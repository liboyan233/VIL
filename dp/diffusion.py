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
from inference_control.srv import GetTrajectory_umi, GetTrajectory_umiResponse
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace

gripper_scale = 1000  # switch from m to mm

class DiffusionServer:
    def __init__(self, ckpt_path, normalizer_path):
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
        #     self.config = pickle.load(f)
        self.service = rospy.Service('/get_trajectory', GetTrajectory_umi, self.handle_request)
        rospy.loginfo("Service is ready.")


    def handle_request(self, req):
        self.policy.reset()
        with torch.no_grad():
            obs_dict_np = self.get_obs_dict(req)
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            result = self.policy.predict_action(obs_dict)
            action = result['action_pred'][0].detach().to('cpu').numpy()
            action_pose10d = action[..., :9]
            action_grip = action[..., 9:]
            action_pose = mat_to_pose(pose10d_to_mat(action_pose10d))
        gripper_trajectory = JointTrajectory()
        robot_trajectory = JointTrajectory()
        print('-'*20)
        for i in range(len(action_grip)):
            gripper_point = JointTrajectoryPoint()
            gripper_point.positions = action_grip[i].tolist()
            gripper_trajectory.points.append(gripper_point)
    
            robot_point = JointTrajectoryPoint()
            robot_point.positions = action_pose[i].tolist()
            robot_trajectory.points.append(robot_point)
            
            print(f"Robot positions: {robot_point.positions}")
            print(f"Gripper positions: {gripper_point.positions}")
            
        return GetTrajectory_umiResponse(gripper_trajectory=gripper_trajectory, robot_trajectory=robot_trajectory)


    def get_obs_dict(self, req):
        bridge = cv_bridge.CvBridge()
        wrist_view = bridge.imgmsg_to_cv2(req.images[0], desired_encoding="passthrough")
        side_view = bridge.imgmsg_to_cv2(req.images[1], desired_encoding="passthrough")

        wrist_img = self.process_img(wrist_view)  # (C, H, W)
        side_img = self.process_img(side_view)   # (C, H, W)
        wrist_img = wrist_img[np.newaxis, ...]
        side_img = side_img[np.newaxis, ...]
        # wrist_img = np.repeat(wrist_img[np.newaxis, ...], 2, axis=0)  # (2, C, H, W)
        # side_img = np.repeat(side_img[np.newaxis, ...], 2, axis=0)    # (2, C, H, W)

        states = np.array(req.states, dtype=np.float32)
        print(f"Received states: {states.shape}")
        states[:6] = 0
        robot_state = mat_to_pose10d(pose_to_mat(states[:6]))[np.newaxis, ...] # (1, 9)
        # robot_state = np.repeat(robot_state, 2, axis=0)  # (2, 9)
        eef_pos = robot_state[..., :3]  # (2, 3)
        eef_rot_axis_angle = robot_state[..., 3:]  # (2, 6)

        start_state = np.array(req.start_state, dtype=np.float32) if req.start_state else None
        if start_state is None:
            raise ValueError("Start state is required for relative pose computation.")
        start_mat = pose_to_mat(start_state[:6])
        states_mat = pose_to_mat(states[:6])
        rela_start_mat = convert_pose_mat_rep(
            states_mat,
            base_pose_mat=start_mat,
            pose_rep='relative',
            backward=False)
        rela_start_pose = mat_to_pose10d(rela_start_mat)[np.newaxis, ...]
        
        gripper_state = states[6] * gripper_scale
        gripper_pos = np.array([[gripper_state]], dtype=np.float32)  # (1, 1)
        # gripper_pos = np.repeat(gripper_pos, 2, axis=0)  # (2, 1)

        obs_dict_np = {
            "camera0_rgb": wrist_img,           # (1, C, H, W)
            "camera1_rgb": side_img,             # (1, C, H, W)
            "robot0_eef_pos": eef_pos,        # (1, 3)
            "robot0_eef_rot_axis_angle": eef_rot_axis_angle,  # (1, 6)
            "robot0_gripper_width": gripper_pos,  # (1, 1)
            "robot0_eef_rot_axis_angle_wrt_start": rela_start_pose[:, 3:]  # (1, 6)
        }
        return obs_dict_np

    def process_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        return img
    
    def state_preprocess(self, states):

        qpos_mean = np.array(self.config["qpos_mean"], dtype=np.float32)
        qpos_mean = np.full(states.shape, qpos_mean, dtype=np.float32)
        qpos_std = np.array(self.config["qpos_std"], dtype=np.float32)
        return (states - qpos_mean) / qpos_std
    
    def state_postprocess(self, states):

        qpos_mean = np.array(self.config["qpos_mean"], dtype=np.float32)
        qpos_mean = np.full(states.shape, qpos_mean, dtype=np.float32)
        qpos_std = np.array(self.config["qpos_std"], dtype=np.float32)
        return states * qpos_std + qpos_mean
    
if __name__ == "__main__":
    rospy.init_node('diffusion_model_server')
    folder = "/home/admin128/Desktop/liboyan/umi/data/outputs/2025.08.19/19.41.24_train_diffusion_unet_timm_umi/"
    ckpt_path = folder + "checkpoints/epoch=0180-train_loss=0.013.ckpt"
    normalizer_path = folder + "normalizer.pkl"
    server = DiffusionServer(ckpt_path=ckpt_path, normalizer_path=normalizer_path)
    rospy.spin()