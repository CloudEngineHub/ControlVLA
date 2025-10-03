import os
import sys
import hydra
from omegaconf import OmegaConf
import pathlib
from loguru import logger
import matplotlib.pyplot as plt
import re
import json
import select
import click

import copy
import dill
import h5py
import time
import torch
import pickle
# import rospy
import numpy as np
import collections
import random
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from typing import Dict, Callable, Tuple, List
from umi.common.pose_util import (
    pose_to_mat, mat_to_pose, 
    mat_to_pose10d, pose10d_to_mat
)
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pose_repr_util import (
    compute_relative_pose, 
    convert_pose_mat_rep
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_timm_policy import DiffusionTransformerTimmPolicy

# from sam2.sam2_workspace import SAM2Workspace
# from examples.cameras import MultiCamera
# from core.sdk_client.astribot_client import Astribot
# from franky_control.realsense_camera_client import RealsenseCamera
from umi.real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_obs_dict_wcontrolobs,)
                                                # get_real_umi_action)
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from real_world.sam2_workspace import SAM2Workspace
# from real_world.sam2_workspace_box import SAM2WorkspaceBox

from umi.real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from multiprocessing.managers import SharedMemoryManager
from real_world.time import Rate
from real_world.gopro_camera import CameraHandler

## >>>> Your Customized Robot Arm and IK Wrapper
from real_world.robot_arm import RobotArm
from real_world.ik_wrapper import IKWrapper
## <<<< Your Customized Robot Arm and IK Wrapper

OmegaConf.register_new_resolver("eval", eval, replace=True)

## publish rate
PUBLISH_RATE = 10

## execution window
EXCUTION_OFFSET = 3
EXCUTION_STEPS = 5

def euler_xyz_to_6d(roll, pitch, yaw):
    # Convert degrees to radians if needed
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # Define the rotation matrix for roll (around the x-axis)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Define the rotation matrix for pitch (around the y-axis)
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Define the rotation matrix for yaw (around the z-axis)
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix using the XYZ convention (R = Rz * Ry * Rx)
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    # Extract the first two columns to form the 6D rotation representation
    r1 = R[:, 0]
    r2 = R[:, 1]
    
    # Flatten and concatenate the columns into a 6D vector
    rot_6d = np.concatenate([r1, r2])
    
    return rot_6d

def tx_robot2policy(robot_tcp_pos, robot_tcp_quat, robot_gripper_width):
    """
    Convert robot pos (xyz) and quat (wxyz) to policy pos (xyz) and rot_vec (xyz)
    """
    robot_tcp_pos = np.array(robot_tcp_pos)
    robot_tcp_quat = np.array(robot_tcp_quat)
    policy_gripper_width = np.array(robot_gripper_width)
    # convert robot pos to policy pos
    policy_tcp_pos = robot_tcp_pos
    # convert robot quat to policy rot_vec
    r = R.from_quat(robot_tcp_quat)
    policy_tcp_rot_vec = r.as_rotvec()
    return policy_tcp_pos, policy_tcp_rot_vec, policy_gripper_width

def tx_policy2robot(policy_tcp_pos, policy_tcp_rot_vec, policy_gripper_width):
    """
    Convert policy pos (xyz) and rot_vec (xyz) to robot pos (xyz) and quat (wxyz)
    """
    policy_tcp_pos = np.array(policy_tcp_pos)
    policy_tcp_rot_vec = np.array(policy_tcp_rot_vec)
    robot_gripper_width = np.array(policy_gripper_width)
    # convert policy pos to robot pos
    robot_tcp_pos = policy_tcp_pos
    # convert policy rot_vec to robot quat
    r = R.from_rotvec(policy_tcp_rot_vec)
    robot_tcp_quat = r.as_quat()
    return robot_tcp_pos, robot_tcp_quat, robot_gripper_width

def get_real_umi_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='abs'
    ):

    n_robots = int(action.shape[-1] // 10)
    env_action = list()
    for robot_idx in range(n_robots):
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            env_obs[f'robot{robot_idx}_eef_pos'][-1],
            env_obs[f'robot{robot_idx}_eef_rot_axis_angle'][-1]
        ], axis=-1))

        start = robot_idx * 10
        action_pose10d = action[..., start:start+9]
        action_grip = action[..., start+9:start+10]
        action_pose_mat = pose10d_to_mat(action_pose10d)

        # solve relative action
        action_mat = convert_pose_mat_rep(
            action_pose_mat, 
            base_pose_mat=pose_mat,
            pose_rep=action_pose_repr,
            backward=True)

        # convert action to pose
        action_pose = mat_to_pose(action_mat)
        env_action.append(action_pose)
        env_action.append(action_grip)

    env_action = np.concatenate(env_action, axis=-1)
    return env_action

class RealRobotInference():
    def __init__(self, model, sam2_workspace, cfg_model):
    
        self.model = model
        self.cfg_model = cfg_model
        self.sam2_workspace = sam2_workspace
        self.device = torch.device('cuda')
        self.obs_pose_repr = self.cfg_model.task.pose_repr.obs_pose_repr
        self.action_pose_repr = self.cfg_model.task.pose_repr.action_pose_repr

        self.prepare_camera()
        print(f'Camera connected')
        # replace <ROBOTARM_ARGS> with your robot arm arguments
        self.robot = RobotArm(<ROBOTARM_ARGS>)
        self.ik_wrapper = IKWrapper(<IKWRAPPER_ARGS>)
        time.sleep(3)

        # prepare observation deque
        self.observation_window = deque(maxlen=2)

        # wait for the camera ready
        img_obs = self.get_camera_observation()
        # print(f'Camera image shape: {img_obs.shape}')

        # init episode start pose
        self.current_robot_qpos = None
        self.update_observation_window()
        current_obs = copy.deepcopy(self.observation_window[-1])
        self.episode_start_pose = [
            np.concatenate([
                current_obs[f'robot0_eef_pos'],
                current_obs[f'robot0_eef_rot_axis_angle']
            ], axis=-1)
        ]

        # warm up the model
        print(f'Warming up the model...')
        time.sleep(1)
        self.update_observation_window()
        for _ in range(2):
            robot_action_tcp_pos, robot_action_tcp_quat, robot_action_gripper_width = self.inference_fn(warmup=True)

    def prepare_camera(self):
        v4l_paths = get_sorted_v4l_paths(by_id=False)
        v4l_path = v4l_paths[0]
        self.camera = CameraHandler(v4l_path)
        self.camera.start()
        time.sleep(2)
        print(f'Connected to camera: {v4l_path}')

    def get_camera_observation(self):
        image = self.camera.get_latest_image()
        image = np.array(image / 255., dtype=np.float32)
        return image

    def update_observation_window(self, ):
        current_robot_state = self.robot.get_latest_state()
        current_robot_tcp_pos = current_robot_state['tcp_pos']
        current_robot_tcp_quat = current_robot_state['tcp_quat']
        current_robot_gripper_width = current_robot_state['gripper_width']
        current_image = self.get_camera_observation()
        polict_tcp_pos, policy_tcp_rot_vec, policy_gripper_width = tx_robot2policy(current_robot_tcp_pos, current_robot_tcp_quat, current_robot_gripper_width)
        self.current_robot_qpos = current_robot_state['qpos']

        self.observation_window.append(
            {
                'robot0_eef_pos': polict_tcp_pos,
                'robot0_eef_rot_axis_angle': policy_tcp_rot_vec,
                'robot0_gripper_width': np.array([policy_gripper_width]),
                'camera0_rgb': current_image,
            }
        )
    
    def get_obs(self) -> dict:
        assert len(self.observation_window) >= 2, 'window length < 2'
        # concatenate the last two observations
        obs = {}
        for key in self.observation_window[0].keys():
            obs[key] = np.stack([self.observation_window[0][key], self.observation_window[1][key]], axis=0)
        return obs
    
    def inference_fn(self, warmup=False):
        # >>> pre-process observation (check image dim rgborbgr)
        obs = self.get_obs()
        obs_dict_np, control_obs_dict_np = get_real_umi_obs_dict_wcontrolobs(
            env_obs=obs, shape_meta=self.cfg_model.task.shape_meta, 
            obs_pose_repr=self.obs_pose_repr,
            episode_start_pose=self.episode_start_pose
        )
        raw_image = obs_dict_np['camera0_rgb_narrow'][-1]
        pred_masks = self.sam2_workspace.predict(raw_image)
        if warmup:
            # save the concat (image, mask) for warmup, 
            # always use the last image and all mask
            import cv2
            raw_image = np.transpose((raw_image * 255).astype(np.uint8), (1, 2, 0))
            masks = np.concatenate([mask[0] for mask in pred_masks.values()], axis=1)
            masks = (masks * 255).astype(np.uint8)
            import PIL.Image as Image
            pil_image = Image.fromarray(raw_image)
            # save pil image
            pil_image.save(os.path.join('data/eval/raw.png'))
            # cv2.imwrite('data/eval/raw.png', raw_image)
            cv2.imwrite('data/eval/masks.png', masks)

        control_obs_dict_np['camera0_rgb_narrow_objs'] = []
        for obj_id, mask in pred_masks.items():
            control_obs_dict_np['camera0_rgb_narrow_objs'].append(mask)
        control_obs_dict_np['camera0_rgb_narrow_objs'] = np.stack(control_obs_dict_np['camera0_rgb_narrow_objs'], axis=0).squeeze(1)
        obs_dict = dict_apply(obs_dict_np, 
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
        control_obs_dict = dict_apply(control_obs_dict_np,
            lambda x: torch.from_numpy(x.astype(np.float64)).unsqueeze(0).to(self.device))
        # <<< pre-process observation

        # >>> model inference
        result = self.model.predict_action(obs_dict=obs_dict, control_obs_dict=control_obs_dict,
                                    text=self.cfg_model.task.dataset.language_condition)
        raw_action = result['action_pred'][0].detach().cpu().numpy()
        # <<< model inference

        # >>> post-process action
        policy_action = get_real_umi_action(raw_action, obs, self.action_pose_repr)  # action pose repr as rel
        robot_tcp_pos, robot_tcp_quat, robot_gripper_width = [], [], []
        for i_act in policy_action:
            i_policy_tcp_pos = i_act[:3]
            i_policy_tcp_rot_vec = i_act[3:6]
            i_policy_gripper_width = i_act[6]
            i_robot_tcp_pos, i_robot_tcp_quat, i_robot_gripper_width = tx_policy2robot(i_policy_tcp_pos, i_policy_tcp_rot_vec, i_policy_gripper_width)
            robot_tcp_pos.append(i_robot_tcp_pos)
            robot_tcp_quat.append(i_robot_tcp_quat)
            robot_gripper_width.append(i_robot_gripper_width)
        robot_tcp_pos = np.array(robot_tcp_pos)
        robot_tcp_quat = np.array(robot_tcp_quat)
        robot_gripper_width = np.array(robot_gripper_width)
        # <<< post-process action
        return robot_tcp_pos, robot_tcp_quat, robot_gripper_width
    
    def inference(self):
        with torch.inference_mode(), \
            KeystrokeCounter() as key_counter:
            t = 0
            rate = Rate(PUBLISH_RATE)

            while True:
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        exit(0)
                    elif key_stroke == KeyCode(char='h'):
                        # Home & Exit program
                        self.robot.move_to_home()
                        exit(0)
                st = time.time()
                robot_action_tcp_pos, robot_action_tcp_quat, robot_action_gripper_width = self.inference_fn()
                print(f'Inference time: {time.time() - st}')
                for i_step in range(EXCUTION_STEPS):
                    i_pos, i_quat, i_gripper_width = robot_action_tcp_pos[i_step + EXCUTION_OFFSET], robot_action_tcp_quat[i_step + EXCUTION_OFFSET], robot_action_gripper_width[i_step + EXCUTION_OFFSET]
                    i_rotmat = R.from_quat(i_quat).as_matrix()
                    i_target_qpos = self.ik_wrapper.inverse_kinematics(i_rotmat, i_pos.tolist(), self.current_robot_qpos.tolist())

                    print(f'exe gripper_width: {i_gripper_width}')
                    self.robot.send_qpos(i_target_qpos, i_gripper_width, async_move=True)
                    self.update_observation_window()
                    rate.sleep()
                print(f'Published Step: {t}')
                t += 1


@click.command()
@click.option('--input_path', '-i', required=True, help='Path to checkpoint')
@click.option('--prompt_path', '-p', required=True, help='Path to prompt')
def main(input_path, prompt_path):
    # load checkpoint
    ckpt_path = input_path
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    with open(ckpt_path, 'rb') as f:
        payload = torch.load(f, map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    
    # configure model
    model: DiffusionTransformerTimmPolicy
    model = hydra.utils.instantiate(cfg.policy)
    model.load_state_dict(payload['state_dicts']['model'])

    # action following evaluation
    model.to('cuda')
    model.eval()
    device = model.device

    # load sam2 workspace
    sam2_annopath = prompt_path
    logger.info(f'loading sam2 workspace from {sam2_annopath}')
    sam2_workspace = SAM2Workspace()
    sam2_workspace.init_predictor(sam2_annopath)
    # sam2_workspace = SAM2WorkspaceBox()
    # sam2_workspace.init_predictor(sam2_annopath)

    env_eval = RealRobotInference(model, sam2_workspace, cfg)
    env_eval.inference()

if __name__ == '__main__':
    main()

