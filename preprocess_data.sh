#! /bin/bash
DATASET="kitti360"
SEQ_ID="4950"
STATIC_SEMANTIC_PATH="/root/docker_data/Lid-Lab-NeRF/0000004916_0000005264.ply"
DYNAMIC_SEMANTIC_PATH="/root/docker_data/Lid-Lab-NeRF/0000004916_0000005264_dynamic.ply"

python -m data.preprocess.generate_rangeview_with_image --dataset $DATASET --sequence_id $SEQ_ID --static_semantic_data_path $STATIC_SEMANTIC_PATH --dynamic_semantic_data_path $DYNAMIC_SEMANTIC_PATH

python -m data.preprocess.kitti360_to_nerf_with_image --sequence_id $SEQ_ID --static_semantic_data_path $STATIC_SEMANTIC_PATH --dynamic_semantic_data_path $DYNAMIC_SEMANTIC_PATH

python -m data.preprocess.cal_seq_config --dataset $DATASET --sequence_id $SEQ_ID