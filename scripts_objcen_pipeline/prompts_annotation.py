# %%
import sys
import os
import zarr
import click

# ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
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
@click.option('-i', '--inputfolder', required=True, help='Project directory')
def main(inputfolder):
    # read the zarr.zip file
    zarr_file_path = os.path.join(inputfolder, f'{os.path.dirname(inputfolder).split("/")[-1]}.zarr.zip')
    with zarr.ZipStore(zarr_file_path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )
    cmd = 0
    while True:
        camera0_rgb_narrow = replay_buffer.data['camera0_rgb_narrow']
        init_image = camera0_rgb_narrow[int(cmd)]
        # visualize the first image
        import matplotlib.pyplot as plt
        plt.imshow(init_image)
        plt.show()
        # take the cmd input
        cmd = input("Enter the frame index/'q':")
        if cmd == 'q':
            break

if __name__ == '__main__':
    main()