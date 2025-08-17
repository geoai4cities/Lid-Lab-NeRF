This repository is the official PyTorch implementation for Lid-Lab-NeRF.

## Introduction  

Lid-Lab-NeRF is pytorch-based NeRF framwork to generate novel scans for LiDAR pointclouds (from previously unexplored sensor locations). Lid-Lab-NeRF provides a wholistic approach to synthetic LiDAR data generation, it generates depth, intensity, raydrop and semantic labels altogether. Lid-Lab-NeRF also uses a post-processing pipeline that generates raydrop patterns to match the real data. Along with this our framework also captures the movement of dynamic object through scans to generate position accurate, non distorted outputs. Our framework is the first work that does multi-property prediction along with capturing the dynamic objects and generating realistic scans, in a signle combined setup. 



### Installation

```bash
git clone https://github.com/Kafka2122/Lid-Lab-NeRF.git
cd Lid-Lab-NeRF

conda create -n labelnerf python=3.9
conda activate labelnerf

# PyTorch
# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA <= 11.7
# pip install torch==2.0.0 torchvision torchaudio

# Dependencies
pip install -r requirements.txt

# Local compile for tiny-cuda-nn
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn/bindings/torch
python setup.py install

# compile packages in utils
cd utils/chamfer3D
python setup.py install
```


### Dataset

#### KITTI-360 dataset ([Download](https://www.cvlibs.net/datasets/kitti-360/download.php))
We use sequence00 (`2013_05_28_drive_0000_sync`) for experiments in our paper.   

Download KITTI-360 dataset (2D images are not needed) and put them into `data/kitti360`. Download the data3D semantics and search for the semantic file name that matches
your sequence, for example for `2013_05_28_drive_0000_sync` file name `0000004916_0000005264_dynamic.ply` and `0000004916_0000005264.ply` are the 3D semantic files. The semantic files are labelled as `{start_frame}_{endframe}.ply` so you need to pick these files accordingly so that all the scans in your sequence folder fall in between the timeframe `start_frame` and `end_frame` and the corresponding labels are availble.  

The folder tree is as follows:  

```bash
data
└── kitti360
    └── KITTI-360
        ├── calibration
        ├── data_3d_raw
        └── data_poses
```

Next, run KITTI-360 dataset preprocessing: (set `DATASET`, `SEQ_ID`, `STATIC_SEMANTIC_PATH` and `DYNAMIC_SEMANTIC_PATH` (the 3d semantic files you downloaded))  

```bash
bash preprocess_data.sh
```

After preprocessing, your folder structure should look like this:  

```bash
configs
├── kitti360_{sequence_id}.txt
data
└── kitti360
    ├── KITTI-360
    │   ├── calibration
    │   ├── data_3d_raw
    │   └── data_poses
    ├── train
    ├── transforms_{sequence_id}test.json
    ├── transforms_{sequence_id}train.json
    └── transforms_{sequence_id}val.json
```

### Run Lid-Lab-NeRF

Set corresponding sequence config path in `--config` and you can modify logging file path in `--workspace`. Remember to set available GPU ID in `CUDA_VISIBLE_DEVICES`.   
Run the following command:
```bash
# KITTI-360
bash run_kitti_labelnerf.sh
```

<!-- <a id="simulation"></a> -->

## Simulation


After the training is completed, you can use the simulator to render and manipulate LiDAR point clouds in the whole scenario. It supports dynamic scene re-play, novel LiDAR configurations (`--fov_lidar`, `--H_lidar`, `--W_lidar`) and novel trajectory (`--shift_x`, `--shift_y`, `--shift_z`).  
We also provide a simple demo setting to transform LiDAR configurations from KITTI-360 to NuScenes, using `--kitti2nus` in the bash script. 
In the `main_labelnerf_sim.py` you can also check line 267 to simulate a sine trajectory for the simulated sensor.
To generate denser scans increase `--H_lidar` and `--W_lidar` to something like 128 and 2048 respectively. 
Check the sequence config and corresponding workspace and model path (`--ckpt`).  
Run the following command:
```bash
bash run_kitti_labelnerf_sim.sh
```
The results will be saved in the workspace folder.


## Acknowledgement
We sincerely appreciate the great contribution of the following works:
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/tree/master)
- [LiDAR4D](https://github.com/ispc-lab/LiDAR4D)
- [LiDAR-NeRF](https://github.com/tangtaogo/lidar-nerf)
- [NeRF-LiDAR](https://github.com/fudan-zvg/NeRF-LiDAR)
- [NFL](https://research.nvidia.com/labs/toronto-ai/nfl/)
- [K-Planes](https://github.com/sarafridov/K-Planes)



## License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
