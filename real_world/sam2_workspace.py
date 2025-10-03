import sys
import os
import pickle
import copy
from typing import Dict, Optional
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil
from collections import OrderedDict

from PIL import Image
import hydra
from hydra import initialize, compose

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize

class SAM2Workspace:
    def __init__(self, config:str='sam2_hiera_t.yaml', 
                    checkpoint:str='./data/checkpoints/sam2_hiera_tiny.pt',
                    device='cuda'):
        # Initialize Hydra's global config
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        initialize(config_path="./configs")  # Update this path
        self.cfg = config
        # Build the SAM2 video predictor
        self.predictor = build_sam2_video_predictor(self.cfg, checkpoint, device=device)
        self.device = device
        self.predictor_image_size = self.predictor.image_size

    @torch.no_grad()
    def init_predictor(self, initanno_path: str, 
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
                async_loading_frames=False,):
        # load the initial annotations
        initanno = pickle.load(open(initanno_path, 'rb'))
        # if initanno is a dict, convert it to a list
        if isinstance(initanno, dict):
            initanno = [initanno]
        self.num_reference_frames = len(initanno)
        init_images = []
        for initanno_i in initanno:
            init_image = self.normalize_image(initanno_i['frame'], self.predictor_image_size)
            init_images.append(init_image)
        init_images = torch.stack(init_images, dim=0)
        ## form init inference_state dict
        inference_state = {
            'images': init_images,
            'num_frames': self.num_reference_frames,
            'offload_video_to_cpu': offload_video_to_cpu,
            'offload_state_to_cpu': offload_state_to_cpu,
            'video_height': 224,
            'video_width': 224,
            'device': self.device,
            'storage_device': self.device,
            'point_inputs_per_obj': {},
            'mask_inputs_per_obj': {},
            'cached_features': {},
            'constants': {},
            'obj_id_to_idx': OrderedDict(),
            'obj_idx_to_id': OrderedDict(),
            'obj_ids': [],
            'output_dict': {
                'cond_frame_outputs': {},
                'non_cond_frame_outputs': {},
            },
            'output_dict_per_obj': {},
            "temp_output_dict_per_obj": {},
            'consolidated_frame_inds': {
                "cond_frame_outputs": set(),  # set containing frame indices
                "non_cond_frame_outputs": set(),  # set containing frame indices
            },
            'tracking_has_started': False,
            'frames_already_tracked': {},
        }
        # warmup the predictor
        self.inference_state = inference_state
        for i_frame in range(self.num_reference_frames):
            self.predictor._get_image_feature(self.inference_state, frame_idx=i_frame, batch_size=1)
        self.predictor.reset_state(self.inference_state)
        # config the mask prompts
        for i_frame in range(self.num_reference_frames):
            initanno_i = initanno[i_frame]
            try:
                prompts = initanno_i['prompts']
            except KeyError:
                print('prompts not found, try pormpts instead(typo...)')
                prompts = initanno_i['pormpts']
            for obj_id in prompts:
                points = np.array(prompts[obj_id]["points"], dtype=np.float32)
                labels = np.array(prompts[obj_id]["labels"], np.int32)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=i_frame, # 0
                    obj_id=obj_id, # 1
                    points=points,
                    labels=labels,
                )
        # process the first annotation frame
        self.predictor.propagate_in_video_preflight(self.inference_state)
        self.output_dict = self.inference_state['output_dict']
        self.obj_ids = self.inference_state['obj_ids']
        self.batch_size = self.predictor._get_obj_num(self.inference_state)
        clear_non_cond_mem = self.predictor.clear_non_cond_mem_around_input and (
            self.predictor.clear_non_cond_mem_for_multi_obj or self.batch_size <= 1
        )
        storage_key = "cond_frame_outputs"
        current_out = self.output_dict[storage_key][0]
        pred_masks = current_out["pred_masks"]
        if clear_non_cond_mem:
            # clear non-conditioning memory of the surrounding frames
            for i_frame in range(self.num_reference_frames):
                self.predictor._clear_non_cond_mem_around_input(inference_state, i_frame)
        for i_frame in range(self.num_reference_frames):
            self.predictor._add_output_per_object(
                    inference_state, i_frame, current_out, storage_key)
        for i_frame in range(self.num_reference_frames):
            self.inference_state["frames_already_tracked"][i_frame] = {"reverse": False}

    def normalize_image(self,
        image: np.ndarray,
        image_size: int,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),) -> torch.Tensor:
        """Normalize an image, (H, W, C) -> (C, H, W)."""
        image = image.astype(np.float32)
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        if image.max() > 1.2:
            image /= 255.0
        image -= img_mean
        image /= img_std
        image = image.transpose(2, 0, 1)
        return torch.tensor(image, device=self.device)

    def _clear_state(self):
        self.inference_state['images'] = self.inference_state['images'][:self.num_reference_frames]
        self.inference_state['num_frames'] = self.num_reference_frames
        self.inference_state['output_dict']['non_cond_frame_outputs'] = {}
        self.inference_state["cached_features"] = {}
        for obj_id in range(len(self.inference_state['obj_ids'])):
            self.inference_state['output_dict_per_obj'][obj_id]['non_cond_frame_outputs'] = {}
            self.inference_state['temp_output_dict_per_obj'][obj_id]['non_cond_frame_outputs'] = {}
        self.inference_state['frames_already_tracked'] = {i_frame: {"reverse": False} for i_frame in range(self.num_reference_frames)}

    @torch.no_grad()
    def predict(self, raw_image: np.ndarray):
        # clear state
        self._clear_state()
        # predict
        self.inference_state["num_frames"] += 1
        image = self.normalize_image(raw_image, self.predictor.image_size)
        self.inference_state['images'] = torch.cat([self.inference_state['images'], image.unsqueeze(0)], dim=0)
        storage_key = "non_cond_frame_outputs"
        current_out, pred_masks = self.predictor._run_single_frame_inference(
            inference_state=self.inference_state,
            output_dict=self.output_dict,
            # frame_idx=frame_idx,  # 1
            frame_idx=self.num_reference_frames,
            batch_size=self.batch_size,  # 1
            is_init_cond_frame=False,
            point_inputs=None,
            mask_inputs=None,
            reverse=False,
            run_mem_encoder=True,
        )
        self.output_dict[storage_key][self.num_reference_frames] = current_out
        self.predictor._add_output_per_object(
            self.inference_state, self.num_reference_frames, current_out, storage_key
        )
        self.inference_state["frames_already_tracked"][self.num_reference_frames] = {"reverse": False}
        _, video_res_masks = self.predictor._get_orig_video_res_output(
            self.inference_state, pred_masks
        )
        objs_segment = {
            obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
            for i, obj_id in enumerate(self.obj_ids)
        }
        return objs_segment

if __name__ == '__main__':
    # Create the workspace instance
    sam2 = SAM2Workspace()
    sam2.init_predictor('./example_finetune_demo/picknplace_toy.d10/picknplace_toy.d10.objectcentric.anno.pkl')
    
    for i in tqdm(range(1, 100)):
        imagepath = f'./example_finetune_demo/cache/frames/{i:05d}.jpg'
        raw_image = cv2.imread(imagepath)
        objs_segment = sam2.predict(raw_image)
        if i == 50:
            from PIL import Image
            # show raw image and segment
            raw_image = Image.fromarray(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
            raw_image.show()
            for obj_id, mask in objs_segment.items():
                mask = Image.fromarray(mask[0])
                mask.show()
            pass
