#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python main_labelnerf_sim.py \
--config configs/kitti360_4950.txt \
--workspace log/kitti360_labelnerf_f4950_release/simulation \
--ckpt log/kitti360_labelnerf_f4950_release/checkpoints/labelnerf_ep0639_refine.pth \
--fov_lidar 2.0 26.9 \
--H_lidar 66 \
--W_lidar 1030 \
--shift_x 0.0 \
--shift_y 0.0 \
--shift_z 0.0 \
--align_axis \
--use_refine True \
# --kitti2nus
