from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
from loguru import logger

def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()

def pose_distance(start_pose, end_pose):
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    return np.linalg.norm(start_pose - end_pose)

class AngleInterpolation:
    OFFSETS = np.array([0, 0, 0, 0, 0, -1.5, 0])  ## offset the joint6 angle by -1 radian

    def __init__(self, times, qposes):
        self.times = times
        self.qposes = qposes + self.OFFSETS
        self.slerps = []
        for dim in range(len(qposes[0])):
            rot = st.Rotation.from_euler('xyz', [[qpose[dim], 0, 0] for qpose in self.qposes])
            try:
                self.slerps.append(st.Slerp(times, rot))
            except:
                # logger.error(f'AngleInterpolation __init__: {times}')
                # logger.error(f'AngleInterpolation __init__: {qposes}')
                raise

    def __call__(self, t):
        try:
            poses =  np.array([slerp(t).as_euler('xyz')[0] for slerp in self.slerps]).reshape(-1, 7)
            return poses - self.OFFSETS
        except:
            raise
    
    @property
    def x(self):
        return self.slerps[0].times
    
    @property
    def y(self):
        poses = []
        for t in self.x:
            poses.append(self(t))
        return np.concatenate(poses, axis=0)

class QPoseTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        assert len(times) >= 1
        try:
            assert len(poses) == len(times)
        except:
            raise
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)

        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            self.qpos_interp = AngleInterpolation(times, poses)
    
    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.qpos_interp.x
    
    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        else:
            return self.qpos_interp.y
            

    def trim(self, 
            start_t: float, end_t: float
            ) -> "QPoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # remove duplicates, Slerp requires strictly increasing x
        all_times = np.unique(all_times)
        # interpolate
        all_poses = self(all_times)
        # logger.info(all_poses.shape)
        # if len(all_poses) == 2:
        #     print()
        return QPoseTrajectoryInterpolator(times=all_times, poses=all_poses)
    
    def drive_to_waypoint(self, 
            pose, time, curr_time,
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf
        ) -> "QPoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        time = max(time, curr_time)
        
        curr_pose = self(curr_time)
        dist = pose_distance(curr_pose, pose)
        duration = time - curr_time
        duration = max(dist / max_pos_speed, duration)
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # insert new pose
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = QPoseTrajectoryInterpolator(times, poses)
        return final_interp

    def schedule_waypoint(self,
            pose, time, 
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf,
            curr_time=None,
            last_waypoint_time=None
        ) -> "QPoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        if last_waypoint_time is not None:
            assert curr_time is not None

        # trim current interpolator to between curr_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                # logger.warning(f'insert time {time} is earlier than current time {curr_time}, no effect done to the interpolator')
                return self
            # now, curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        # end time should be the latest of all times except time
        # after this we can assume order (proven by zhenjia, due to the 2 min operations)

        # Constraints:
        # start_time <= end_time <= time (proven by zhenjia)
        # curr_time <= start_time (proven by zhenjia)
        # curr_time <= time (proven by zhenjia)
        
        # time can't change
        # last_waypoint_time can't change
        # curr_time can't change
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        # after this, all waypoints in trimmed_interp is within start_time and end_time
        # and is earlier than time

        # determine speed
        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        dist = pose_distance(pose, end_pose)
        # duration = dist / max_pos_speed
        duration = max(duration, dist / max_pos_speed)
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new pose
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        try:
            poses = np.append(trimmed_interp.poses, [pose], axis=0)
        except:
            poses = np.append(trimmed_interp.poses, [pose], axis=0)
            raise

        # create new interpolator
        final_interp = QPoseTrajectoryInterpolator(times, poses)
        return final_interp


    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])
        
        pose = np.zeros((len(t), 7))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)
            for i, t_ in enumerate(t):
                pose[i] = self.qpos_interp(t_)
            

        if is_single:
            pose = pose[0]
        # logger.info(pose.shape)
        # if pose.shape == (1, 7):
            # print()
        return pose
