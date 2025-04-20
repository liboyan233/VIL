import h5py
import cv2
import os 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from PIL import Image
"""
[config]
solved: 1 (0?)
[data]
 --- arm & ee ---
Data key: ctrl_arm, value: (N, 7)
Data key: ctrl_ee, value: (N, 1)
Data key: qp_arm, value: (N, 7) ---*
Data key: qp_ee, value: (N,) ---*
Data key: qv_arm, value: (N, 7)
Data key: qv_ee, value: (N,)
 --- depth & rgb ---
Data key: d_left, value: (N, 240, 424) 
Data key: d_right, value: (N, 240, 424)
Data key: d_top, value: (N, 240, 424)
Data key: d_wrist, value: (N, 240, 424)
Data key: rgb_left, value: (N, 240, 424, 3) ---*
Data key: rgb_right, value: (N, 240, 424, 3) ---*
Data key: rgb_top, value: (N, 240, 424, 3) ---*
Data key: rgb_wrist, value: (N, 240, 424, 3) ---*

Data key: time, value: (N,)
"""
NAN_SIZE = 2 # 2 attached nan for action

def load_data_online(dataset_dir, batch_size_train, batch_size_val, transform=None, camera_cfg=None):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    # train_ratio = 0.8
    # shuffled_indices = np.random.permutation(num_episodes)
    # train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    # val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    if camera_cfg is None:
        train_cam = ['rgb_left', 'rgb_top']
        eval_cam = ['rgb_right', 'rgb_top']
    else:
        train_cam = camera_cfg['train_camera_names']
        eval_cam = camera_cfg['test_camera_names']
    # construct dataset and dataloader
    train_dataset = RoboSetDataset(dataset_dir, train_cam, norm_stats=None, transform=transform)
    val_dataset = RoboSetDataset(dataset_dir, eval_cam, train_dataset.norm_stats, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, train_dataset.norm_stats

class RoboSetDataset(Dataset):
    def __init__(self, path, img_choice=['rgb_left', 'rgb_right', 'rgb_top', 'rgb_wrist'], norm_stats=None,transform=None):
        self.base_path = path
        self.trial_num, self.trial_list, self.file_list = self.trial_count(path)
        self.episode_len = self.eps_len(path)
        self.img_choice = img_choice
        self.transform = transform
        if norm_stats is None:
            self.norm_stats = self.get_norm_stats(path)
        else:
            self.norm_stats = norm_stats
    
    def __len__(self):
        return self.trial_num
    
    def __getitem__(self, idx):
        file_id, trial_id = self.get_data_ids(idx)
        file_name = self.file_list[file_id]

        episode_len = self.episode_len[idx]
        start_ts = np.random.choice(episode_len)
        image_data, qpos_data, action_data, is_pad = self.datareader(file_name, trial_id, start_ts)

        return image_data, qpos_data, action_data, is_pad

    @staticmethod
    def visualize(rgb):
        cv2.imshow('image', rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_data_ids(self, idx): # return file id and trial id
        for i in range(len(self.trial_list)):
            if idx < self.trial_list[0]:
                return 0, idx
            elif idx < self.trial_list[i]:
                return i, idx - self.trial_list[i - 1]
        raise ValueError("Index out of range")

    @staticmethod
    def datareader_all(file_path):
        h5 = h5py.File(file_path, 'r')
        print(h5.keys())
        for step, key in enumerate(h5.keys()):
            print(h5[f'Trial{step}'].keys())
            trial = h5[f'Trial{step}']

            config_group = trial['config']
            config_key = 'solved'
            config_value = config_group[config_key][()]
            print(f"Config key: {config_key}, value: {config_value}")
            
            data_group = trial['data']
            print(f"Data group keys: {data_group.keys()}")
            for data_key in data_group.keys():
                data_value = data_group[data_key][()]
                print(f"Data key: {data_key}, value: {data_value.shape}")
            return data_group

    def datareader(self, file_name, trial_id, start_ts):
        file_path = os.path.join(self.base_path, file_name)
        h5 = h5py.File(file_path, 'r')
        # print('keys ', h5.keys())
        trail_key = list(h5.keys())[trial_id]
        trial = h5[trail_key]

        data_seq = {}
        act_arm = trial['data']['ctrl_arm'][()]
        act_ee = trial['data']['ctrl_ee'][()]
        data_seq['action'] = np.concatenate((act_arm, act_ee), axis=1) # N, 8
        qp_arm = trial['data']['qp_arm'][()]
        qp_ee = trial['data']['qp_ee'][()]
        qp_ee = qp_ee.reshape(-1, 1)
        data_seq['qp'] = np.concatenate((qp_arm, qp_ee), axis=1)

        data_seq['rgb_left'] = trial['data']['rgb_left'][()] # N, 240, 424, 3
        data_seq['rgb_right'] = trial['data']['rgb_right'][()]
        data_seq['rgb_top'] = trial['data']['rgb_top'][()]
        data_seq['rgb_wrist'] = trial['data']['rgb_wrist'][()]

        image_data, qpos_data, action_data, is_pad = self.seq_segment(data_seq, start_ts)

        return image_data, qpos_data, action_data, is_pad
    
    def seq_segment(self, data_seq, start_ts):
        episode_len = data_seq['action'].shape[0] - NAN_SIZE
        act_nonpad = data_seq['action'][max(0, start_ts - 1):-NAN_SIZE]
        action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned
        padded_action = np.zeros_like(data_seq['action'], dtype=np.float32)
        padded_action[:action_len] = act_nonpad

        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1
        qpos = data_seq['qp'][start_ts]

        imgs = []
        for cam in self.img_choice:
            
            if self.transform is not None:
                img_pil = Image.fromarray(data_seq[cam][start_ts])
                img = self.transform(img_pil)
                img = np.array(img)
                imgs.append(img)
            else:
                imgs.append(data_seq[cam][start_ts])
        all_cam_images = np.stack(imgs, axis=0) # 4, 240, 424, 3

                # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad
    
    def trial_count(self, path):
        trial_num = 0
        trial_list = []
        file_list = []
        for _, _, files in os.walk(path):
            for file in files:
                if file.endswith('.h5'):
                    if '20230306' in file:
                        print('skip ', file)
                        continue
                    file_list.append(file)
                    file_path = os.path.join(path, file)
                    h5 = h5py.File(file_path, 'r')
                    keys = list(h5.keys())
                    lens = len(keys)
                    h5.close()
                    trial_num += lens
                    trial_list.append(lens)
        print(f"Total trial number: {trial_num}")
        trial_list = np.cumsum(np.array(trial_list))
        print(f"Trial list: {trial_list}")
        print(f"File list: {file_list}")
        return trial_num, trial_list, file_list

    def eps_len(self, path):
        eps_len_list = []
        for file in self.file_list:
            file_path = os.path.join(path, file)
            h5 = h5py.File(file_path, 'r')
            for key in h5.keys():
                trial = h5[key]
                eps_len = trial['data']['ctrl_arm'].shape[0]
                eps_len_list.append(eps_len)
        return eps_len_list
    
    def get_norm_stats(self, path):
        all_qpos_data = []
        all_action_data = []
        for file in self.file_list:
            file_path = os.path.join(path, file)
            h5 =  h5py.File(file_path, 'r')
            for key in h5.keys():
                trial = h5[key]
                qp_arm = trial['data']['qp_arm'][()]
                qp_ee = trial['data']['qp_ee'][()]
                qp_ee = qp_ee.reshape(-1, 1)
                qpos = np.concatenate((qp_arm, qp_ee), axis=1)

                act_arm = trial['data']['ctrl_arm'][()]
                act_ee = trial['data']['ctrl_ee'][()]
                action = np.concatenate((act_arm, act_ee), axis=1)

                all_qpos_data.append(torch.from_numpy(qpos))
                all_action_data.append(torch.from_numpy(action))

        all_qpos_data = torch.stack(all_qpos_data).to(torch.float32)
        all_action_data = torch.stack(all_action_data).to(torch.float32)

        all_action_data = all_action_data[:, :-NAN_SIZE] # remove nan
        # normalize action data
        # print('shape check', all_action_data.shape)
        action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
        action_std = all_action_data.std(dim=[0, 1], keepdim=True)
        action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

        # normalize qpos data
        qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
        qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

        stats = {
                "action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
                "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
                }

        return stats
        

if __name__ == '__main__':
    path = '/home/data/liboyan/dataset/RoboSet/source/baking_prep_slide_open_drawer_scene_1/test/'
    robo_dataset = RoboSetDataset(path)
    dataloader = DataLoader(robo_dataset, batch_size=25, shuffle=False)

    for i, data in enumerate(dataloader):
        # image_data, qpos_data, action_data, is_pad
        print('data shape ', data[0].shape, data[1].shape, data[2].shape, data[3].shape)
        # print('demo action: ', data[2][0][:5])
        # print('demo qpos: ', data[1][0])
        # for j in range(data[0].shape[0]):
        #     RoboSetDataset.visualize(data[0][j][0].permute(1, 2, 0).numpy())
        #     print('i ', i, 'j ', j)
        #     if j > 0:
        #         break
        # if i == 2:
        #     break