import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
from deoxys.utils.ik_utils import IKWrapper
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.qpose_trajectory_interpolator import QPoseTrajectoryInterpolator
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait
from umi.common.pose_util import pose_to_mat, mat_to_pose
from loguru import logger
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface as FrankaInterfaceBase
from deoxys.utils import YamlConfig
from deoxys.utils import transform_utils
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
PER_FRAME_POS_DELTA_THRESHOLD = 0.003
PER_FRAME_ROT_DELTA_THRESHOLD = 0.03
GRIPPER_WIDTH = 0.085 # approx 7.7cm

def normalize_gripper_width(gripper_width):
    # For FrankaInterface.control, -1 means opening the gripper and 0 means close.
    # But for FrankaInterface.last_gripper_q, it's the opening width in meters.
    if gripper_width is None:
        logger.warning(f'gripper_width is None')
        return -0.5
    return - np.clip(gripper_width / GRIPPER_WIDTH, 0, 1) 

def format_output_array(array):
    if isinstance(array, np.ndarray):
        return np.round(array, 5).tolist()
    elif isinstance(array, list):
        return [format_output_array(item) for item in array]
    elif isinstance(array, dict):
        return {key: format_output_array(value) for key, value in array.items()}
    else:
        return array

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


tx_flangerot90_tip = np.identity(4)
# tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.247])
tx_flangerot90_tip[:3, 3] = np.array([0., 0., 0.])

tx_flange_tip = tx_flangerot90_tip
tx_tip_flange = np.linalg.inv(tx_flange_tip)

class FrankaInterface(FrankaInterfaceBase):
    def __init__(self, *args, **kwargs):
        # FrankaInterfaceBase.__init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

    def get_ee_pose(self):
        if self.state_buffer_size == 0:
            return None
        O_T_EE = np.array(self._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()
        flange_pose =  np.concatenate([O_T_EE[:3, 3], transform_utils.mat2euler(O_T_EE[:3, :3])])
        tip_pose = mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)
        return tip_pose
    
    def get_joint_positions(self):
        print(f'[WARN  ] FrankaInterface.move_to_joint_positions not implemented')
    
    def get_joint_velocities(self):
        print(f'[WARN  ] FrankaInterface.move_to_joint_positions not implemented')

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        print(f'[WARN  ] FrankaInterface.move_to_joint_positions not implemented')

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        print(f'[WARN  ] FrankaInterface.start_cartesian_impedance not implemented')
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        print(f'[WARN  ] FrankaInterface.update_desired_ee_pose not implemented')

    def terminate_current_policy(self):
        pass

    def close(self):
        super().close()

# _DEBUG_GLOBAL_STEP = 0

class FrankaDeoxysController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """
    def __init__(self,
        shm_manager: SharedMemoryManager, 
        # robot_ip,
        # robot_port=4242,
        frequency=1000,
        Kx_scale=1.0,
        Kxd_scale=1.0,
        launch_timeout=3,
        joints_init=None,
        joints_init_duration=None,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0,
        simulation:bool=False,
        ):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="FrankaPositionalController")
        self.robot_ip = 'TO_BE_SET'
        self.interface_cfg = os.path.join(config_root, 'charmander.yml')
        self.controller_cfg = None
        self.frequency = frequency
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        if os.environ.get('DEBUG'):
            self.verbose = True
        else:
            self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'target_gripper_width': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd','get_joint_velocities'),
            ('gripper_position', 'get_gripper_position'),
            ('last_eef_posrpy', 'last_eef_posrpy')
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(7)
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)
            elif 'gripper_position' in func_name:
                example[key] = 0.0
            elif 'last_eef_posrpy' in key:
                example[key] = np.zeros(6)
            # elif 'gripper_velocity' in func_name:
            #     example[key] = 0.0

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        example['gripper_receive_timestamp'] = time.time()
        example['gripper_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
        self.simulation = simulation
            
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        logger.info(f'Franka Deoxys Controller stop')
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (7,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose[:6],
            'target_gripper_width': pose[6],
            'target_time': target_time
        }
        self.input_queue.put(message)
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

    def get_gripper_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
            
        robot = FrankaInterface(self.interface_cfg, use_visualizer=True, 
                                control_freq=self.frequency,
                                state_freq=100, automatic_gripper_reset=True,
                                simulation=self.simulation)
        time.sleep(3)
        try:
            if self.verbose:
                print(f"[FrankaPositionalController] Connect to robot: {self.robot_ip}")
            
            ik_wrapper = IKWrapper()
            if os.environ.get('DEBUG'):
                self.ready_event.set()
            
            # main loop
            dt = 1. / self.frequency
            time.sleep(1) # this sleep ensures that the eef_posrpy and last_gripper_q thread has enough time to update the values.
            curr_pose = robot.last_eef_posrpy
            curr_qpose = robot.last_q
            if curr_pose is None:
                logger.error(f'Franka Deoxys Controller returned None pose, consider "activate FCI"')
                raise ValueError
            else:
                logger.debug(f'Franka Deoxys Controller initial pose: {np.round(curr_pose, 5).tolist()}')

            curr_gripper_width = normalize_gripper_width(robot.last_gripper_q)
            logger.debug(f'Franka Deoxys Controller initial gripper width: {np.round(curr_gripper_width, 5)}')

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            qpose_interp = QPoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_qpose]
            )
            target_gripper_width = curr_gripper_width
            
            gripper_width_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_gripper_width,0,0,0,0,0]]
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            controller_type = 'JOINT_IMPEDANCE'
            controller_cfg = YamlConfig(config_root + f"/joint-impedance-controller.yml").as_easydict()
            # controller_cfg = YamlConfig(config_root + f"/compliant-joint-impedance-controller.yml").as_easydict()
            # controller_type = 'JOINT_POSITION'
            # controller_cfg = YamlConfig(config_root + f"/joint-position-controller.yml").as_easydict()
            # controller_type = 'OSC_POSE'
            # controller_cfg = YamlConfig(config_root + f"/osc-pose-controller.yml").as_easydict()
            print(f'controller_cfg: {controller_cfg}')

            target_pose = np.zeros(6)
            # last_pos = robot.last_eef_posrpy
            desired_q_pose = curr_qpose
            _DEBUG_GLOBAL_STEP = 0
            while keep_running:
                t_now = time.monotonic()
                # target_gripper_width = gripper_width_interp(t_now)[0:1]
                # if iter_idx % 5 == 0:
                #     logger.info(f'q target delta:                          {np.round(np.array(robot.last_q) - np.array(desired_q_pose), 5).tolist()}')
                #     logger.info(f'pos delta:                               {np.round(np.array(robot.last_eef_posrpy) - np.array(target_pose), 5)}')
                # logger.debug(f'target_pose: {np.round(target_pose, 5).tolist()}')
                # logger.info(f'velocity: {np.sqrt(np.sum(np.square(np.array(robot.last_eef_posrpy) - np.array(last_pos)))) * 20}')
                # last_pos = robot.last_eef_posrpy
                # target_gripper_width = -1
                # desired_q_pose = qpose_interp(t_now)

                target_gripper_width = gripper_width_interp(t_now)[0]
                target_desired_q_pose = qpose_interp(t_now)
                # target_desired_q_pose = desired_q_pose
                action = target_desired_q_pose.tolist() + [target_gripper_width]
                # _DEBUG_GLOBAL_STEP += 1
                # FREQ = 100
                # if (_DEBUG_GLOBAL_STEP // FREQ) % 2 == 0:
                #     i_step = (_DEBUG_GLOBAL_STEP // 10) % FREQ
                #     action = [0.0, 
                #               0., 
                #               0., 
                #               0., 
                #               0., 
                #               0.,
                #               0.5] \
                #         + [-1.0]
                # else:
                #     action = [0.0, 
                #               0., 
                #               0., 
                #               0., 
                #               0., 
                #               0.,
                #               0.5] \
                #         + [-1.0]

                # if 'last_action' not in locals():
                #     last_action = action.copy()
                #     logger.info(f'q action: {np.round(action, 5).tolist()}')
                # else:
                #     if np.abs(np.array(action) - np.array(last_action)).max() > 0.02:
                #         logger.info(f'q action: {np.round(action, 5).tolist()}')
                #         last_action = action.copy()
                # if iter_idx % 5 == 0:
                # logger.info(f'q action: {np.round(action, 5).tolist()}')
                    # logger.info(f'q pose :  {np.round(robot.last_q).tolist()}')
                assert np.array(action).shape == (8, ), f'Wrong shape: {np.array(action).shape}'
                # time.sleep(0.4)
                logger.warning("control loop...")
                logger.warning(f'q action: {np.round(action, 5).tolist()}')
                robot.control(
                    controller_type=controller_type,
                    controller_cfg=controller_cfg,
                    action=action)
                # logger.info(f'curr os pose  : {np.round(robot.last_eef_posrpy, 5).tolist()}')
                # logger.info(f'current q pose: {np.round(robot.last_q, 5).tolist()}')
                state = dict()
                for key, func_name in self.receive_keys:
                    if key == 'ActualTCPPose':
                        flange_pose = robot.last_eef_posrpy
                        # tip_pose = mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)
                        # state[key] = tip_pose
                        state[key] = np.zeros(6)
                        state[key][:3] = flange_pose[:3]
                        state[key][3:6] = st.Rotation.from_euler('xyz', flange_pose[3:6]).as_rotvec()
                        # state[key][3:6] = flange_pose[3:6]
                    elif key == 'ActualQ':
                        state[key] = robot.last_q
                    elif key == 'ActualQd':
                        state[key] = robot.last_q_d
                    elif key == 'gripper_position':
                        state[key] = robot.last_gripper_q
                    elif key == 'last_eef_posrpy':
                        state[key] = robot.last_eef_posrpy
                    # elif key == 'gripper_velocity':
                    #     state[key] = robot.last_gripper
                    # state[key] = getattr(robot, func_name)()

                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['gripper_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                state['gripper_receive_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])

                except Empty:
                    n_cmd = 0
                    commands = None

                for i in range(n_cmd):
                    # logger.debug(f'{n_cmd=}')
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.STOP.value:
                        keep_running = False
                        logger.info(f'Franka Deoxys Controller received stop command')
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        raise NotImplementedError
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        print(f'[UserWarning] FrankaDeoxysController __run__ SERVOL, not implement gripper action')
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        qpose_interp = qpose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[FrankaPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        # logger.debug(f'Franka Deoxys Controller schedule waypoint: {command}')
                        target_pose = command['target_pose']
                        # logger.info(f'target pose: {target_pose}')
                        
                        target_mat = np.eye(4)
                        target_mat[:3, :3] = st.Rotation.from_euler('xyz', target_pose[3:6]).as_matrix()
                        # target_mat[:3, :3] = st.Rotation.from_euler('xyz', robot.last_eef_posrpy[3:6]).as_matrix()
                        target_mat[:3, 3] = target_pose[:3]
                        target_rotation = target_mat[:3, :3]
                        target_position = target_mat[:3, 3].reshape(-1)

                        desired_q_pose  = ik_wrapper.inverse_kinematics(target_rotation, target_position, robot.last_q.tolist())
                        logger.info(f'desired q pose: {np.round(desired_q_pose, 5).tolist()}')
                        logger.info(f'current q pose: {np.round(robot.last_q, 5).tolist()}')
                        logger.info(f'target os pose: {np.round(target_pose, 5).tolist()}')
                        logger.info(f'current os pose: {np.round(robot.last_eef_posrpy, 5).tolist()}\n')

                        target_gripper_width = command['target_gripper_width'][0]
                        target_time = float(command['target_time'])
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = time.monotonic()
                        qpose_interp = qpose_interp.schedule_waypoint(
                            pose=desired_q_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        gripper_width_interp = gripper_width_interp.schedule_waypoint(
                            pose=[target_gripper_width, 0, 0, 0, 0, 0],
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        logger.info(f'Franka Deoxys Controller received unknown command: {cmd} {command}')
                        keep_running = False
                        break

                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util,slack_time=0, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                if self.verbose:
                    # print('dt', dt)
                    # logger.info(f"[FrankaPositionalController] Actual frequency {1/(time.monotonic() - t_now)}")
                    pass

            logger.critical(f'End of run')
        except Exception as e:
            logger.critical(f'Exception {e}')
            import traceback
            print(traceback.format_exc())
            raise e
        finally:
            # manditory cleanup
            # terminate
            logger.critical('terminate_current_policy')
            robot.terminate_current_policy()
            robot.close()
            del robot
            self.ready_event.set()

            if self.verbose:
                logger.info(f"[FrankaPositionalController] Disconnected from robot: {self.robot_ip}")


if __name__ == '__main__':
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    this_robot = FrankaDeoxysController(
        shm_manager=shm_manager,
        frequency=200,
        Kx_scale=1.0,
        Kxd_scale=np.array([2.0,1.5,2.0,1.0,1.0,1.0]),
        verbose=False,
        receive_latency=0.0001,

        # joints_init=[
        #     0.09162008114028396,
        #     -0.19826458111314524,
        #     -0.01990020486871322,
        #     -2.4732269941140346,
        #     -0.01307073642274261,
        #     2.30396583422025,
        #     0.8480939705504309,
        # ]
    )
    this_robot.run()
