# Zero-Shot View-Invariant Imitation Learning: Bridging Handheld Gripper and Any Exocentric View
![Architecture Diagram](sources/overall.png "System Overview")
### Description: 
This repo contains the implementation of VIL, Zero-Shot View-Invariant Imitation Learning framework.<br>
A large part of codes are adopted from ACT algorithm. We would like to thank the authors of ACT for their well-organized project!<br>
In this repo, you can run VIL in simulation environments such as Transfer Cube and Bimanual Insertion for training and evaluation. You can also test the effect of different encoder backbone and SOTA methods like DP. <br>
This work has been submitted to ICRA 2026. The hardware / real world deployment part is coming soon.

### Installation
The installation has been tested under Ubuntu 22.04, CUDA 12.1
```bash
conda create -n vil python=3.8.10
conda activate vil
pip install -r requirements.txt
cd act/detr && pip install -e .
cd ..
```
### Simulated experiments
We use ``sim_transfer_cube_scripted`` task in the examples below. Another option is ``sim_insertion_scripted``.
To generate 50 episodes of scripted data, run:
```python
python3 record_sim_episodes.py \
--task_name sim_transfer_cube_scripted \
--dataset_dir <data save dir> \
--num_episodes 50
```
we have do modifications to generate right/left/top/wrist views at same time.

You can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the episode after it is collected, run
    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

To train ACT:
```python
# Transfer Cube task
python -W ignore imitate_episodes.py --task_name  sim_transfer_cube_scripted --ckpt_dir <./ckpt_save_path> --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0
```
To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

Please refer to [act](https://github.com/tonyzhaozh/act) for more training tips

### Test different backbones & different camera viewpoints
In constants.py, one may alter the settings of:
        'train_camera_names': ['wrist', 'left'], # 'wrist', 'left', 'top'
        'test_camera_names': ['wrist', 'left'] # 'wrist', 'right', 'top'
train_camera_names refers to the input views for training. test_camera_names refers to the input views for validation & eval. You may use different views during training/validation/evaluation
Currently the used backbone list are resnet18 + swin-tiny, set in imitate_episodes.py 
```python
backbone = ['resnet18', 'swin_tiny']
```
one may use alternative 'resnet18' or 'swin_tiny' or 'vit_b_16'
Note that the order of image processing in the backbone corresponds exactly to the order defined in train_camera_names and test_camera_names
### Comare with ACT baseline
You may set backbones to all resnet18 for ACT baseline
### Compare with Diffusion Policy
We adapt the evaluation and dataset generation for comparison with Diffusion Policy (DP).
please follow [umi](https://github.com/real-stanford/universal_manipulation_interface) to config the python environments and download the diffusion_policy folder in that repo to ./diffusion_policy
after the hdf5 files generate above, one may run scripts under dp_dataset to transfer to zarr.zip format under ./dp_dataset/ 
```python
python act1_dataset_transfer.py -i </path/to/hdf5/folder> -o <zarr file_name1> # for cube transfer
python act2_dataset_transfer.py -i </path/to/hdf5/folder> -o <zarr file_name2> # for insertion
```
Afterwards, one may do training with DP with minor modifications on task yaml files, we provide a demo yaml file in ./dp_dataset/transfer_cube.yaml.
we further provide a demo script to evaluate dp within the above simulation environment:
```python
python -W ignore eval_dp.py  --task_name sim_transfer_cube_scripted  --ckpt_dir /path/to/<some_dp_ckpt>.ckpt    --normalizer_path /path/to/dp/normalizer.pkl --policy_class DP --seed 0 --eval
```
### Hand-held gripper models
[CAD Models](https://drive.google.com/drive/folders/15jRhqiBxu612BTisAuTE6C8DEEFC9AZC?usp=drive_link)

<!-- ### Data acquisition & processing
comming soon -->

<!-- ### Training & Eval demos
comming soon -->
