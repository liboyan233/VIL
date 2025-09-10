import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import zarr
import pickle
import numpy as np
import cv2
import av
import multiprocessing
import concurrent.futures
import h5py
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from collections import defaultdict
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
import argparse
register_codecs()

"""
action (Dataset): shape=(400, 14), dtype=float32 -- jointspace
observations/ (Group)
observations/images/ (Group)
observations/images/left (Dataset): shape=(400, 480, 640, 3), dtype=uint8
observations/images/right (Dataset): shape=(400, 480, 640, 3), dtype=uint8
observations/images/top (Dataset): shape=(400, 480, 640, 3), dtype=uint8
observations/images/wrist (Dataset): shape=(400, 480, 640, 3), dtype=uint8
observations/qpos (Dataset): shape=(400, 14), dtype=float32 -- jointspace
observations/qvel (Dataset): shape=(400, 14), dtype=float32 -- noneeded
"""

def get_image_transform(in_res, out_res, crop_ratio:float = 1.0, bgr_to_rgb: bool=False):
    iw, ih = in_res
    ow, oh = out_res
    ch = round(ih * crop_ratio)
    cw = round(ih * crop_ratio / oh * ow)
    interp_method = cv2.INTER_AREA

    w_slice_start = (iw - cw) // 2
    w_slice = slice(w_slice_start, w_slice_start + cw)
    h_slice_start = (ih - ch) // 2
    h_slice = slice(h_slice_start, h_slice_start + ch)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        assert img.shape == ((ih,iw,3))
        # crop
        img = img[h_slice, w_slice, c_slice]
        # resize
        img = cv2.resize(img, out_res, interpolation=interp_method)
        return img
    
    return transform

def main(args):
    output = args.output
    in_res = args.in_res
    out_res = args.out_res
    input_path = args.input_dir

    input_files = list(pathlib.Path(input_path).glob('*.hdf5'))
    print(f"Found {len(input_files)} input files in {input_path}")

    if os.path.isfile(output):
        response = input(f'Output file {output} exists! Overwrite? [y/N]: ')
        if response.lower() not in ('y', 'yes'):
            print("Aborted.")
            return

    out_res = tuple(int(x) for x in out_res.split(','))
    in_res = tuple(int(x) for x in in_res.split(','))

    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    resize_tf = get_image_transform(
        in_res=in_res,
        out_res=out_res
    )
    for h5_path in input_files:
        h5_path = pathlib.Path(h5_path).absolute()
        try:
            with h5py.File(h5_path, 'r') as f:
                obs = f['observations']
                wrist_imgs = obs['images']['wrist'][()]
                wrist2_imgs = obs['images']['wrist2'][()]
                side_imgs = obs['images']['left'][()]
                n_frames = side_imgs.shape[0]
    
                wrist_resize_imgs = np.empty((n_frames, *out_res, 3), dtype=np.uint8)
                wrist2_resize_imgs = np.empty((n_frames, *out_res, 3), dtype=np.uint8)
                side_resize_imgs = np.empty((n_frames, *out_res, 3), dtype=np.uint8)
                for i in range(n_frames):
                    wrist_resize_imgs[i] = resize_tf(wrist_imgs[i])
                    wrist2_resize_imgs[i] = resize_tf(wrist2_imgs[i])
                    side_resize_imgs[i] = resize_tf(side_imgs[i])

                qpos = obs['qpos'][()]
                action = f['action'][()]
                action = action[:n_frames]

                assert action.shape[1] == 14, f"Expected action shape (N, 14), got {action.shape}"
                assert qpos.shape[1] == 14, f"Expected qpos shape (N, 14), got {qpos.shape}"
                
                print(f"Processing {h5_path}, action shape: {action.shape}, wrist_imgs shape: {wrist_resize_imgs.shape}, side_imgs shape: {side_resize_imgs.shape}")

                episode_data = {
                    'robot_qpos': action[..., ].astype(np.float32),
                    'action': action[..., ].astype(np.float32),
                    'camera0_rgb': wrist_resize_imgs[:n_frames],
                    'camera1_rgb': wrist2_resize_imgs[:n_frames],
                    'camera2_rgb': side_resize_imgs[:n_frames],
                }
                
                out_replay_buffer.add_episode(data=episode_data, compressors=None)
                
        except OSError as e:
            print(f"❌ 无法打开文件: {h5_path}，原因: {e}")
            continue
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer ACT dataset to Zarr format.")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Input HDF5 file(s) path')
    parser.add_argument('-o', '--output', required=True, help='Zarr path')
    parser.add_argument('-or', '--out_res', type=str, default='224,224', help='Output resolution, e.g. 224,224')
    parser.add_argument('-ir', '--in_res', type=str, default='640,480', help='Input resolution, e.g. 960,540')
    parser.add_argument('-t', '--task', type=str, default='cube_transfer', help='Task name')
    args = parser.parse_args()
    main(args)