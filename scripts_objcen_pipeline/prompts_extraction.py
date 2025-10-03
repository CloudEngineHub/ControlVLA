# %%
import sys
import os
import zarr
import click
import pickle

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
# try to read the .zarr.zip file, extract the data and save it again
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

# %%
register_codecs()

# %%
@click.command()
@click.option('-i', '--inputtask', required=True, help='Project directory')
def main(inputtask):
    # read the zarr.zip file
    zarr_file_path = os.path.join('example_finetune_demo', inputtask, f'{inputtask}.zarr.zip')
    with zarr.ZipStore(zarr_file_path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )
    camera0_rgb_narrow = replay_buffer.data['camera0_rgb_narrow']

    if inputtask == 'picknplace_toy.d10':
        annotation = [
            {
                'frame': camera0_rgb_narrow[0],
                'prompts': {
                    1: {"points": [[77, 163], [77, 154]], 
                        "labels": [1, 1]}, ## 1 green toy
                    2: {"points": [[43, 132], ], 
                        "labels": [1]}, ## 2 blue bowl
                    },
            },
            {
                'frame': camera0_rgb_narrow[300],
                'prompts': {
                    1: {"points": [[120, 182], [113, 139]], 
                        "labels": [1, 1]}, ## 1 green toy
                    2: {"points": [[28, 145], ], 
                        "labels": [1]}, ## 2 blue bowl
                    },
            },
        ]
    else:
        raise ValueError("The task is not annotated yet.")
    pickle.dump(annotation, open(f"example_finetune_demo/{inputtask}/{inputtask}.objectcentric.anno.pkl", "wb"))

if __name__ == '__main__':
    main()