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

# %%
register_codecs()
# Register object codec (this ensures that object arrays can be serialized)
# import numcodecs
# numcodecs.register_codec(numcodecs.JSON(), 'json')

import click
# read the zarr.zip file
@click.command()
@click.option('-i', '--inputfolder', required=True, help='Project directory')
def main(inputfolder):
    inputtask = inputfolder.split('/')[-1]
    zarr_file_path = os.path.join(inputfolder, f'{inputtask}.zarr.zip')
    target_file_path = os.path.join(inputfolder, f'{inputtask}.objectcentric.zarr.zip')
    with zarr.ZipStore(zarr_file_path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )

    # dump object-centric result
    sam2_result = pickle.load(open(os.path.join(inputfolder, 'sam2_results.pkl'), "rb"))
    num_objs = len(sam2_result['video_segments'][0])
    print(f'number of objects: {num_objs}')
    
    # %%
    name = 'camera0_rgb_narrow_objs'
    out_res = (num_objs, 224, 224)
    _ = replay_buffer.data.require_dataset(
        name=name,
        shape=(replay_buffer['camera0_rgb_narrow'].shape[0],) + out_res,
        chunks=(1,) + out_res,
        # compressor=img_compressor,
        dtype=bool
    )

    from tqdm import tqdm
    print(len(sam2_result['video_segments']))
    for frame_id in tqdm(range(replay_buffer['camera0_rgb_narrow'].shape[0])):
        image_segments = sam2_result['video_segments'][frame_id]
        for obj_idx, obj_id in enumerate(image_segments):
            segment = image_segments[obj_id]
            replay_buffer[name][frame_id, obj_idx] = np.squeeze(segment)

    print(f"Saving ReplayBuffer to {target_file_path}")
    with zarr.ZipStore(target_file_path, mode='w') as zip_store:
        replay_buffer.save_to_store(
            store=zip_store
        )


if __name__ == '__main__':
    main()