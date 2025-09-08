#!/usr/bin/env python
# from pandas import options
import rospy
import cv_bridge
import numpy as np
import cv2
import hydra
import torch
import dill
import pickle
import os
import sys
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import torch.nn as nn

gripper_scale = 1000  # switch from m to mm

class DiffusionPolicy(nn.Module):
    def __init__(self, ckpt_path, normalizer_path):
        super().__init__()
        payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        self.policy = workspace.model
        if cfg.training.use_ema:
            self.policy = workspace.ema_model
        self.policy.num_inference_steps = 16 # DDIM inference iterations

        # --- Add these lines to load and set the normalizer ---
        with open(normalizer_path, "rb") as f:
            normalizer = pickle.load(f)
        self.policy.set_normalizer(normalizer)
        # ------------------------------------------------------

        self.device = torch.device('cuda')
        self.policy.eval().to(self.device)
        # self.config_path = '/home/admin128/Desktop/liboyan/umi/data/outputs/2025.08.16/23.33.44_train_diffusion_unet_timm_umi/normalizer.pkl'
        # with open(self.config_path, "rb") as f:
        #     self.config = pickle.load(f


    def forward(self, qpos, curr_image):
        self.policy.reset()
        with torch.no_grad():
            obs_dict = {
                "robot_qpos": qpos[np.newaxis, ...],  # (1, 1, 14)
                # "camera0_rgb": curr_image[:, :1],  # (1, 1, C, H, W)
                # "camera1_rgb": curr_image[:, 1:],  # (1, 1, C, H, W)
            }
            for i in range(curr_image.shape[1]):
                obs_dict[f"camera{i}_rgb"] = curr_image[:, i:i+1]
                
            action_dict = self.policy.predict_action(obs_dict)
            action = action_dict['action'].detach().to('cpu').numpy()
        # TODO - post process action if needed
        return action

    def process_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        return img
    
if __name__ == "__main__":
    rospy.init_node('diffusion_model_server')
    folder = "/home/admin128/Desktop/liboyan/umi/data/outputs/2025.08.19/19.41.24_train_diffusion_unet_timm_umi/"
    ckpt_path = folder + "checkpoints/epoch=0180-train_loss=0.013.ckpt"
    normalizer_path = folder + "normalizer.pkl"
    # server = DiffusionServer(ckpt_path=ckpt_path, normalizer_path=normalizer_path)
    # rospy.spin()