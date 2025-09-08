import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from roboset_dataloader import load_data_online
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos
import json

from sim_env import BOX_POSE
from torchvision import transforms
import torchvision.transforms.functional as F
from warnings import warn
import warnings
from colorama import Fore, Style
import IPython
import time
from diffusion import DiffusionPolicy
import cv2
CUDA_LAUNCH_BLOCKING=1
e = IPython.embed
# matplotlib.use('Agg') 

def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    print(f"{Fore.YELLOW}Warning:{Style.RESET_ALL} {message} ({filename}:{lineno})")
warnings.showwarning = custom_showwarning


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    normalizer_path = args['normalizer_path']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'

    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    elif 'roboset' in task_name:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14  # 8 for roboset, 14 for sim_transfer_cube

    camera_config = {
        'train_camera_names': task_config['train_camera_names'] if 'train_camera_names' in task_config else camera_names,
        'test_camera_names': task_config['test_camera_names'] if 'test_camera_names' in task_config else camera_names,
    }

    # print('=============== trial:', args['trial'], '===============')
    config = {
        'trial': args['trial'] if 'trial' in args else None,
        'ckpt_dir': ckpt_dir,
        'normalizer_path': normalizer_path,
        'state_dim': state_dim,
        'real_robot': not is_sim,
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'episode_len': episode_len,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_config': camera_config,
        'stepwisel1': args['stepwisel1'] if 'stepwisel1' in args else False,
    }

    if is_eval:
        results = []
        success_rate, avg_return = eval_bc(config, save_episode=True)
        results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    return 


def make_policy(policy_class, ckpt_path, normalizer_path):
    if policy_class == 'DP':
        policy = DiffusionPolicy(ckpt_path, normalizer_path)
    else:
        raise NotImplementedError
    return policy

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

def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = resize_tf(ts.observation['images'][cam_name])
        curr_image = rearrange(curr_image, 'h w c -> c h w')
        # curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    normalizer_path = config['normalizer_path']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    camera_names = config['camera_config']['test_camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    warn(f"Camera used for eval: {camera_names}", )

    # load policy and stats
    policy = make_policy(policy_class, ckpt_dir, normalizer_path)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_dir}')

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = 48

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 10
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        print(f'================ Rollout {rollout_id} ================')
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []

        glab_flag2 = False
        glab_threshold_open = 0.7
        glab_threshold_close = 0.5
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT


                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = torch.from_numpy(qpos_numpy).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)  # N, C, H, W in [0, 1]

                ### query policy
                if config['policy_class'] == "DP":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                        # print('output act shape: ', all_actions.shape)
                    if temporal_agg:
                        pass
                    else:
                        # raw_action = all_actions[:, t % query_frequency]
                        action_idx = (t % query_frequency) // 1  # 1 for downsample factor
                        raw_action = all_actions[:, action_idx]
                else:
                    raise NotImplementedError

                ### post-process actions
                # action = post_process(raw_action)
                target_qpos = raw_action[0]
                # print('target_qpos:', qpos[-1, -1])
                if qpos[-1, -1] > glab_threshold_open:
                    glab_flag2 = True

                if glab_flag2 and qpos[-1, -1] < 0.2:
                    glab_flag2 = False
                
                if glab_flag2 and qpos[-1, -1] < glab_threshold_close:
                    # print(f'change from {target_qpos[-1]} to 0')
                    target_qpos[-1] = 0

                ### step the environment
                ts = env.step(target_qpos)  # the desired input is of shape (14,)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join('/home/admin128/Desktop/liboyan/Imitation_learning/act/ckpt_test', f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + '.txt'
    with open(os.path.join('/home/admin128/Desktop/liboyan/Imitation_learning/act/ckpt_test', result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--normalizer_path', action='store', type=str, help='normalizer_path', required=True)
    # parser.add_argument('--ckpt', action='store', type=str, help='ckpt_name', default=None)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)

    # parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', required=False)

    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--stepwisel1', action='store_true')

    # parser.add_argument('--loadckpt', action='store', type=str, help='train from pretrained policy ckpt', default=None)

    resize_tf = get_image_transform(
        in_res=(640,480),
        out_res=(224,224)
    )
    
    main(vars(parser.parse_args()))
