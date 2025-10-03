from typing import Dict, Callable, Tuple, List
import numpy as np
import collections
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pose_repr_util import (
    compute_relative_pose, 
    convert_pose_mat_rep
)
from umi.common.pose_util import (
    pose_to_mat, mat_to_pose, 
    mat_to_pose10d, pose10d_to_mat)
from diffusion_policy.model.common.rotation_transformer import \
    RotationTransformer
from umi.real_world.franka_deoxys_controller import normalize_gripper_width
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform, 
    draw_predefined_mask,
    inpaint_tag,
    get_mirror_crop_slices
)
import pathlib
import json

def fisheye_distortion(img):
    raise NotImplementedError

# global narrow_fisheye_converter, resize_tf
# intr_path = pathlib.Path('./example/calibration/gopro_intrinsics_2_7k.json')
# opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open('r')))
# # opencv_intr_dict['DIM'] = np.array([224, 224])
# narrow_fisheye_converter = FisheyeRectConverter(
#     **opencv_intr_dict,
#     out_size=(224, 224),
#     out_fov=85
# )
# # IH, IW = 224, 224
# resize_tf = get_image_transform(
#     in_res=(224, 224),
#     out_res=(2704, 2028),
# )
def fast_fisheye_distortion(img):
    # global narrow_fisheye_converter, resize_tf
    # img = resize_tf(img)
    return img

def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res


def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
    return obs_dict_np

def get_real_umi_obs_dict_wcontrolobs(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        obs_pose_repr: str='abs',
        tx_robot1_robot0: np.ndarray=None,
        episode_start_pose: List[np.ndarray]=None,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    controlobs_dict_np = dict()
    # process non-pose
    obs_shape_meta = shape_meta['obs']
    robot_prefix_map = collections.defaultdict(list)
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs['camera0_rgb']
            if key == 'camera0_rgb_fisheye':
                pass
            elif key == 'camera0_rgb_narrow':
                for ti in range(this_imgs_in.shape[0]):
                    this_imgs_in[ti] = fast_fisheye_distortion(this_imgs_in[ti])
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            # print(f'key: {key}, hi: {hi}, wi: {wi}, ho: {ho}, wo: {wo}')
            # print(f'key: {key}, dtype: {this_imgs_in.dtype}')
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim' and ('eef' not in key):
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
            # handle multi-robots
            ks = key.split('_')
            if ks[0].startswith('robot'):
                robot_prefix_map[ks[0]].append(key)
    
    controlobs_shape_meta = shape_meta.get('control_obs', {})
    for key, attr in controlobs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs['camera0_rgb']
            if key == 'camera0_rgb_fisheye':
                pass
            elif key == 'camera0_rgb_narrow':
                for ti in range(this_imgs_in.shape[0]):
                    this_imgs_in[ti] = fast_fisheye_distortion(this_imgs_in[ti])
            elif key == 'camera0_rgb_narrow_objs':
                continue
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            controlobs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        else:
            raise NotImplementedError

    # generate relative pose
    for robot_prefix in robot_prefix_map.keys():
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            env_obs[robot_prefix + '_eef_pos'],
            env_obs[robot_prefix + '_eef_rot_axis_angle']
        ], axis=-1))

        # solve reltaive obs
        obs_pose_mat = convert_pose_mat_rep(
            pose_mat, 
            base_pose_mat=pose_mat[-1],
            pose_rep=obs_pose_repr,
            backward=False)

        obs_pose = mat_to_pose10d(obs_pose_mat)
        obs_dict_np[robot_prefix + '_eef_pos'] = obs_pose[...,:3]
        obs_dict_np[robot_prefix + '_eef_rot_axis_angle'] = obs_pose[...,3:]
    
    # generate pose relative to other robot
    n_robots = len(robot_prefix_map)
    for robot_id in range(n_robots):
        # convert pose to mat
        assert f'robot{robot_id}' in robot_prefix_map
        tx_robota_tcpa = pose_to_mat(np.concatenate([
            env_obs[f'robot{robot_id}_eef_pos'],
            env_obs[f'robot{robot_id}_eef_rot_axis_angle']
        ], axis=-1))
        for other_robot_id in range(n_robots):
            if robot_id == other_robot_id:
                continue
            tx_robotb_tcpb = pose_to_mat(np.concatenate([
                env_obs[f'robot{other_robot_id}_eef_pos'],
                env_obs[f'robot{other_robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            tx_robota_robotb = tx_robot1_robot0
            if robot_id == 0:
                tx_robota_robotb = np.linalg.inv(tx_robot1_robot0)
            tx_robota_tcpb = tx_robota_robotb @ tx_robotb_tcpb

            rel_obs_pose_mat = convert_pose_mat_rep(
                tx_robota_tcpa,
                base_pose_mat=tx_robota_tcpb[-1],
                pose_rep='relative',
                backward=False)
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            obs_dict_np[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
            obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]

    # generate relative pose with respect to episode start
    if episode_start_pose is not None:
        for robot_id in range(n_robots):        
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                env_obs[f'robot{robot_id}_eef_pos'],
                env_obs[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            
            # get start pose
            start_pose = episode_start_pose[robot_id]
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False)
            
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # obs_dict_np[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

    return obs_dict_np, controlobs_dict_np

def get_real_umi_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        obs_pose_repr: str='abs',
        tx_robot1_robot0: np.ndarray=None,
        episode_start_pose: List[np.ndarray]=None,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    # process non-pose
    obs_shape_meta = shape_meta['obs']
    robot_prefix_map = collections.defaultdict(list)
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs['camera0_rgb']
            if key == 'camera0_rgb_fisheye':
                pass
            elif key == 'camera0_rgb_narrow':
                for ti in range(this_imgs_in.shape[0]):
                    this_imgs_in[ti] = fast_fisheye_distortion(this_imgs_in[ti])
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim' and ('eef' not in key):
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
            # handle multi-robots
            ks = key.split('_')
            if ks[0].startswith('robot'):
                robot_prefix_map[ks[0]].append(key)

    # generate relative pose
    for robot_prefix in robot_prefix_map.keys():
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            env_obs[robot_prefix + '_eef_pos'],
            env_obs[robot_prefix + '_eef_rot_axis_angle']
        ], axis=-1))

        # solve reltaive obs
        obs_pose_mat = convert_pose_mat_rep(
            pose_mat, 
            base_pose_mat=pose_mat[-1],
            pose_rep=obs_pose_repr,
            backward=False)

        obs_pose = mat_to_pose10d(obs_pose_mat)
        obs_dict_np[robot_prefix + '_eef_pos'] = obs_pose[...,:3]
        obs_dict_np[robot_prefix + '_eef_rot_axis_angle'] = obs_pose[...,3:]
    
    # generate pose relative to other robot
    n_robots = len(robot_prefix_map)
    for robot_id in range(n_robots):
        # convert pose to mat
        assert f'robot{robot_id}' in robot_prefix_map
        tx_robota_tcpa = pose_to_mat(np.concatenate([
            env_obs[f'robot{robot_id}_eef_pos'],
            env_obs[f'robot{robot_id}_eef_rot_axis_angle']
        ], axis=-1))
        for other_robot_id in range(n_robots):
            if robot_id == other_robot_id:
                continue
            tx_robotb_tcpb = pose_to_mat(np.concatenate([
                env_obs[f'robot{other_robot_id}_eef_pos'],
                env_obs[f'robot{other_robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            tx_robota_robotb = tx_robot1_robot0
            if robot_id == 0:
                tx_robota_robotb = np.linalg.inv(tx_robot1_robot0)
            tx_robota_tcpb = tx_robota_robotb @ tx_robotb_tcpb

            rel_obs_pose_mat = convert_pose_mat_rep(
                tx_robota_tcpa,
                base_pose_mat=tx_robota_tcpb[-1],
                pose_rep='relative',
                backward=False)
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            obs_dict_np[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
            obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]

    # generate relative pose with respect to episode start
    if episode_start_pose is not None:
        for robot_id in range(n_robots):        
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                env_obs[f'robot{robot_id}_eef_pos'],
                env_obs[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            
            # get start pose
            start_pose = episode_start_pose[robot_id]
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False)
            
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # obs_dict_np[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

    return obs_dict_np

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
        env_action.append(normalize_gripper_width(action_grip))

    env_action = np.concatenate(env_action, axis=-1)
    return env_action


class SAM2ObjectCentricWorkspace:
    def __init__(self, ):
        self.cfg = cfg
        