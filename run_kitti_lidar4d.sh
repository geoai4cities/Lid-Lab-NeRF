#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python main_lidar4d.py \
--config configs/kitti360_4950.txt \
--workspace log/kitti360_lidar4d_f4950_release \
--static_semantic_data_path /root/docker_data/MSthesis/0000004916_0000005264.ply \
--dynamic_semantic_data_path /root/docker_data/MSthesis/0000004916_0000005264_dynamic.ply \
--lr 5e-2 \
--num_rays_lidar 1024 \
--iters 30000 \
--epoch 639 \
--alpha_d 1 \
--alpha_i 0.1 \
--alpha_r 0.01 \
--alpha_l 0.01 \
--out_lidar_dim 3 \
--ckpt scratch
# --test_eval


# --refine
# --test_eval
