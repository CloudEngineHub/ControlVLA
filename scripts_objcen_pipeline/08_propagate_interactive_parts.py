# %%
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

# ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
# try to read the .zarr.zip file, extract the data and save it again
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

import hydra
from sam2.build_sam import build_sam2_video_predictor
from hydra.core.global_hydra import GlobalHydra

# %%
register_codecs()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfolder', required=True, help='Project directory')
parser.add_argument('-v', '--verbose', default=None, help='verbose output mask to check')
args = parser.parse_args()

verbose = args.verbose
verbose_every = None if verbose == None else int(verbose)
verbose = (verbose_every != None)
print(f"verbose: {verbose}, verbose_every: {verbose_every}")
inputfolder = args.inputfolder
inputtask = inputfolder.split('/')[-1]

# read the zarr.zip file
zarr_file_path = os.path.join(inputfolder, f"{inputtask}.zarr.zip")
with zarr.ZipStore(zarr_file_path, mode='r') as zip_store:
    replay_buffer = ReplayBuffer.copy_from_store(
        src_store=zip_store, 
        store=zarr.MemoryStore()
    )

camera0_rgb_narrow = replay_buffer.data['camera0_rgb_narrow']

## config the sam2 model
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
sam2_checkpoint = "./data/checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
global NUM_REFERENCE_FRAMES

## config the annotation prompt
@torch.no_grad()
def init_state(
        init_images: torch.Tensor,  ## shape (C, H, W)
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,):
    """Initialize an inference state."""
    global NUM_REFERENCE_FRAMES
    compute_device = device  # device of the model
    # 
    inference_state = {}
    _, _, video_height, video_width = init_images.shape
    # inference_state["images"] = init_image.unsqueeze(0) # need to be a tensor of shape (1, C, H, W)
    inference_state["images"] = init_images

    inference_state["num_frames"] = NUM_REFERENCE_FRAMES # number of frames in the video, need to be 1
    # whether to offload the video frames to CPU memory
    # turning on this option saves the GPU memory with only a very small overhead
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    # whether to offload the inference state to CPU memory
    # turning on this option saves the GPU memory at the cost of a lower tracking fps
    # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
    # and from 24 to 21 when tracking two objeRcts)
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    # the original video height and width, used for resizing final output scores
    inference_state["video_height"] = 224
    inference_state["video_width"] = 224
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    # inputs on each frame
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    # visual features on a small number of recently visited frames for quick interactions
    inference_state["cached_features"] = {}
    # values that don't change across frames (so we only need to hold one copy of them)
    inference_state["constants"] = {}
    # mapping between client-side object id and model-side object index
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    # A storage to hold the model's tracking results and states on each frame
    inference_state["output_dict"] = {
        "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
    }
    # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
    inference_state["output_dict_per_obj"] = {}
    # A temporary storage to hold new outputs when user interact with a frame
    # to add clicks or mask (it's merged into "output_dict" before propagation starts)
    inference_state["temp_output_dict_per_obj"] = {}
    # Frames that already holds consolidated outputs from click or mask inputs
    # (we directly use their consolidated outputs during tracking)
    inference_state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),  # set containing frame indices
        "non_cond_frame_outputs": set(),  # set containing frame indices
    }
    # metadata for each tracking frame (e.g. which direction it's tracked)
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}
    # Warm up the visual backbone and cache the image feature on frame 0
    for frame_idx in range(NUM_REFERENCE_FRAMES):
        predictor._get_image_feature(inference_state, frame_idx=frame_idx, batch_size=1) # warm up the visual backbone
    return inference_state

def clear_state(inference_state: Dict):
    global NUM_REFERENCE_FRAMES
    inference_state['images'] = inference_state['images'][:NUM_REFERENCE_FRAMES]
    inference_state['num_frames'] = NUM_REFERENCE_FRAMES
    inference_state['output_dict']['non_cond_frame_outputs'] = {}
    inference_state["cached_features"] = {}
    for obj_id in range(len(inference_state['obj_ids'])):
        inference_state['output_dict_per_obj'][obj_id]['non_cond_frame_outputs'] = {}
        inference_state['temp_output_dict_per_obj'][obj_id]['non_cond_frame_outputs'] = {}
    inference_state['frames_already_tracked'] = {frame_idx: {"reverse": False} for frame_idx in range(NUM_REFERENCE_FRAMES)}

@torch.no_grad()
def normalize_image(image: np.ndarray,
    image_size: int,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),) -> torch.Tensor:
    """Normalize an image, (H, W, C) -> (C, H, W)."""
    image = image.astype(np.float32)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image /= 255.0
    image -= img_mean
    image /= img_std
    image = image.transpose(2, 0, 1)
    return torch.tensor(image, device=device)

import torchvision.transforms as transforms
resize = transforms.Resize((predictor.image_size, predictor.image_size))

@torch.no_grad()
def normalize_image_tensor(image: torch.Tensor,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),) -> torch.Tensor:
    """Normalize an image, (H, W, C) -> (C, H, W)."""
    image = image.float().permute(2, 0, 1)
    image /= 255.0
    image = resize(image)
    image -= torch.tensor(img_mean, device=device).view(3, 1, 1)
    image /= torch.tensor(img_std, device=device).view(3, 1, 1)
    return image

@torch.no_grad()
def batch_normalize_image_tensor(
    images: torch.Tensor,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Normalize a batch of images, (N, H, W, C) -> (N, C, H, W)."""
    images = images.permute(0, 3, 1, 2).float()
    images /= 255.0
    images = resize(images)
    images -= torch.tensor(img_mean, device=device).view(1, 3, 1, 1)
    images /= torch.tensor(img_std, device=device).view(1, 3, 1, 1)
    return images

with torch.no_grad():
    object_centric_annotation = pickle.load(
        open(os.path.join(inputfolder, f"{inputtask}.objectcentric.anno.pkl"), "rb"))
    if isinstance(object_centric_annotation, dict):
        init_image = object_centric_annotation['frame']
        prompts = object_centric_annotation['prompts']
        init_image = normalize_image(init_image, predictor.image_size)
        init_image = init_image.unsqueeze(0)
        inference_state = init_state(init_image)
        predictor.reset_state(inference_state)
        ## config the mask prompts
        for obj_id in prompts:
            points = np.array(prompts[obj_id]["points"], dtype=np.float32)
            labels = np.array(prompts[obj_id]["labels"], np.int32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0, # 0
                obj_id=obj_id, # 1
                points=points,
                labels=labels,
            )
    elif isinstance(object_centric_annotation, list):
        NUM_REFERENCE_FRAMES = len(object_centric_annotation)
        init_images = []
        for frame_idx in range(NUM_REFERENCE_FRAMES):
            init_image = object_centric_annotation[frame_idx]['frame']
            init_images.append(normalize_image(init_image, predictor.image_size))
        init_images = torch.stack(init_images, dim=0)
        inference_state = init_state(init_images)
        predictor.reset_state(inference_state)
        for frame_idx in range(NUM_REFERENCE_FRAMES):
            prompts = object_centric_annotation[frame_idx]['prompts']
            for obj_id in prompts:
                points = np.array(prompts[obj_id]["points"], dtype=np.float32)
                labels = np.array(prompts[obj_id]["labels"], np.int32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx, # 0
                    obj_id=obj_id, # 1
                    points=points,
                    labels=labels,
                )

    ## propagate the interactive parts
    start_frame_idx = None
    max_frame_num_to_track = None
    reverse = False
    predictor.propagate_in_video_preflight(inference_state)
    output_dict = inference_state["output_dict"]
    consolidated_frame_inds = inference_state["consolidated_frame_inds"]
    obj_ids = inference_state["obj_ids"]
    # num_frames = inference_state["num_frames"] 
    num_frames = camera0_rgb_narrow.shape[0]
    batch_size = predictor._get_obj_num(inference_state)
    if len(output_dict["cond_frame_outputs"]) == 0:
        raise RuntimeError("No points are provided; please add points first")
    clear_non_cond_mem = predictor.clear_non_cond_mem_around_input and (
        predictor.clear_non_cond_mem_for_multi_obj or batch_size <= 1
    )
    # set start index, end index, and processing order
    if start_frame_idx is None:
        # default: start from the earliest frame with input points
        start_frame_idx = min(output_dict["cond_frame_outputs"])
    if max_frame_num_to_track is None:
        # default: track all the frames in the video
        max_frame_num_to_track = num_frames
    if reverse:
        end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
        if start_frame_idx > 0:
            processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
        else:
            processing_order = []  # skip reverse tracking if starting from frame 0
    else:
        end_frame_idx = min(
            start_frame_idx + max_frame_num_to_track, num_frames - 1
        )
        processing_order = range(start_frame_idx, end_frame_idx + 1) # from 0 to end_frame_idx

    video_segments = {}
    for frame_idx in tqdm(processing_order, desc="propagate in real-time video"): # 0  
        start_time = time.time()
        # print(f'consolidated_frame_inds: {consolidated_frame_inds}')
        if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame_idx]
            pred_masks = current_out["pred_masks"]
            if clear_non_cond_mem:
                # clear non-conditioning memory of the surrounding frames
                predictor._clear_non_cond_mem_around_input(inference_state, frame_idx)
        elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
            storage_key = "non_cond_frame_outputs"
            current_out = output_dict[storage_key][frame_idx]
            pred_masks = current_out["pred_masks"]
        else:
            storage_key = "non_cond_frame_outputs"
            current_out, pred_masks = predictor._run_single_frame_inference(
                inference_state=inference_state,
                output_dict=output_dict,
                # frame_idx=frame_idx,  # 1
                frame_idx=NUM_REFERENCE_FRAMES,
                batch_size=batch_size,  # 1
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=reverse,
                run_mem_encoder=True,
            )
            output_dict[storage_key][frame_idx] = current_out
        # elapsed_time = time.time() - start_time
        # print(f"Frame {frame_idx} processed in {elapsed_time:.4f} seconds.")

        # Create slices of per-object outputs for subsequent interaction with each
        # individual object after tracking.
        predictor._add_output_per_object(
            inference_state, frame_idx, current_out, storage_key
        )
        inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

        # Resize the output mask to the original video resolution (we directly use
        # the mask scores on GPU for output to avoid any CPU conversion in between)
        _, video_res_masks = predictor._get_orig_video_res_output(
            inference_state, pred_masks
        )


        # Append current frame to inference state
        if frame_idx < num_frames - 1:
            clear_state(inference_state)
            image = normalize_image(camera0_rgb_narrow[frame_idx + 1], predictor.image_size)
            next_frame_image = image.unsqueeze(0)
            inference_state['images'] = torch.cat((inference_state['images'], next_frame_image), dim=0)
            inference_state["num_frames"] += 1
        

        # Collect the per-frame segmentation results
        video_segments[frame_idx] = {
            obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
            for i, obj_id in enumerate(obj_ids)
        }
        if verbose:
            if frame_idx % verbose_every == 0:
                cols = 4
                import matplotlib.pyplot as plt
                rgb_image = camera0_rgb_narrow[frame_idx]
                fig, axs = plt.subplots(1, cols, figsize=(5*cols, 5))

                # Display RGB image
                axs[0].imshow(rgb_image)
                axs[0].set_title('RGB Image')
                axs[0].axis('off')

                for i_objid, objid in enumerate(obj_ids):
                    axs[i_objid+1].imshow((video_res_masks[i_objid, 0] > 0.0).cpu().numpy())
                    axs[i_objid+1].set_title(f'Object {objid}')
                    axs[i_objid+1].axis('off')
                # Show the plot
                plt.show()

# save video segments
sam2_results = {
    "video_segments": video_segments,
    "annotation": object_centric_annotation,
}
sam2_results_path = os.path.join(inputfolder, "sam2_results.pkl")
with open(sam2_results_path, "wb") as f:
    pickle.dump(sam2_results, f)