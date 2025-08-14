import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from utils.convert import lidar_to_pano_with_intensities, lidar_to_pano_with_intensities_and_labels
from .kitti360_loader import KITTI360Loader
import re
from pathlib import Path
from pyntcloud import PyntCloud
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti360",
        choices=["kitti360"],
        help="The dataset loader to use.",
    )

    parser.add_argument(
        "--sequence_id",
        type=str, 
        default="4950",
        help="choose start",
    )

    parser.add_argument(
        "--static_semantic_data_path", 
        type=str, 
        default= "0000004916_0000005264.ply",
        help="Path to the semantic data file."
    )
    
    parser.add_argument(
        "--dynamic_semantic_data_path", 
        type=str, 
        default= "0000004916_0000005264_dynamic.ply",
        help="Path to the semantic data file."
    )

    return parser


def LiDAR_2_Pano_KITTI(
    local_points_with_intensities_labels, lidar_H, lidar_W, intrinsics, max_depth=80.0
):
    pano, intensities, labels = lidar_to_pano_with_intensities_and_labels(
        local_points_with_intensities_labels=local_points_with_intensities_labels,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=intrinsics,
        max_depth=max_depth,
    )
    range_view = np.zeros((lidar_H, lidar_W, 4))
    range_view[:, :, 1] = intensities
    range_view[:, :, 2] = pano
    range_view[:, :, 3] = labels
    return range_view

def add_label(pointcloud, semantic_data_static, semantic_data_dynamic):
    # Valid static IDs: sidewalk, road, building, vehicle, vegetation
    valid_static_ids = [8, 7, 11, 26, 21] 
    
    # Get static point coordinates and labels
    semantic_points_static = semantic_data_static[['x', 'y', 'z']].values  # Shape (N, 3)
    semantic_ids_static = semantic_data_static['semantic'].values  # Shape (N,)
    
    # Initialize all static labels as 100
    modified_semantic_static = np.full_like(semantic_ids_static, 100)
    
    # Keep only the valid IDs
    for valid_id in valid_static_ids:
        modified_semantic_static[semantic_ids_static == valid_id] = valid_id
    
    # Change ID 34 to building (11)
    modified_semantic_static[semantic_ids_static == 34] = 11
    
    # Set all dynamic points as vehicles (26)
    semantic_ids_dynamic = np.full_like(semantic_data_dynamic['semantic'].values, 26)
    semantic_points_dynamic = semantic_data_dynamic[['x', 'y', 'z']].values
    
    # Create KD-Tree from semantic points
    semantic_tree_static = cKDTree(semantic_points_static)
    semantic_tree_dynamic = cKDTree(semantic_points_dynamic)
    
    # Prepare array for semantic IDs
    semantic_ids_for_pointcloud = np.empty(len(pointcloud), dtype=semantic_ids_static.dtype)
    
    # Find closest points and assign labels
    for i in tqdm(range(len(pointcloud))):
        point = pointcloud[i, :3]  # Get the (x, y, z) part of pointcloud row
        static_dist, static_idx = semantic_tree_static.query(point)
        dynamic_dist, dynamic_idx = semantic_tree_dynamic.query(point)
        
        if static_dist <= dynamic_dist:
            semantic_ids_for_pointcloud[i] = modified_semantic_static[static_idx]
        else:
            semantic_ids_for_pointcloud[i] = semantic_ids_dynamic[dynamic_idx]
    
    # Add semantic labels as new column
    pointcloud_with_semantics = np.hstack((pointcloud, semantic_ids_for_pointcloud.reshape(-1, 1)))
    
    return pointcloud_with_semantics

def generate_train_data(
    H,
    W,
    intrinsics,
    lidar_paths,
    # image_path,
    # semantic_path,
    out_dir,
    points_dim,
    kitti_360_root,
    sequence_name,
    semantic_file_static, 
    semantic_file_dynamic
):
    """
    Args:
        H: Heights of the range view.
        W: Width of the range view.
        intrinsics: (fov_up, fov) of the range view.
        out_dir: Output directory.
        points_dim: Dimensions of each LiDAR point.
    """
    
    # Create the main output directory if it doesn’t exist
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for lidar, image, and semantic_map within the output directory
    # lidar_out_dir = out_dir / 'lidar'
    # image_out_dir = out_dir / 'image'
    # semantic_out_dir = out_dir / 'semantic_map'
    
    # lidar_out_dir.mkdir(parents=True, exist_ok=True)
    # image_out_dir.mkdir(parents=True, exist_ok=True)
    # semantic_out_dir.mkdir(parents=True, exist_ok=True)

    semantic_data_static = PyntCloud.from_file(str(Path(semantic_file_static))).points
    semantic_data_dynamic = PyntCloud.from_file(str(Path(semantic_file_dynamic))).points
    k3 = KITTI360Loader(kitti_360_root)
    # Process LiDAR data and save to 'lidar' directory
    for lidar_path in tqdm(lidar_paths, desc="Processing LiDAR"):
        frame_id = int(re.search(r'(\d+)\.bin$', lidar_path).group(1))
        lidar2world = k3.load_lidars(sequence_name, [frame_id])
        point_cloud = np.fromfile(lidar_path, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, points_dim))
        # Add homogeneous coordinate for transformation
        point_cloud_homogeneous = np.concatenate([point_cloud[:, :3], 
                                                  np.ones((point_cloud.shape[0], 1))], 
                                                  axis=1)
        point_cloud_transformed = (point_cloud_homogeneous @ lidar2world[0].T)

        # Restore intensity values after transformation
        point_cloud_transformed[:, 3] = point_cloud[:, 3]
        
        points_with_label = add_label(pointcloud=point_cloud_transformed,
                              semantic_data_static=semantic_data_static,
                              semantic_data_dynamic = semantic_data_dynamic[semantic_data_dynamic["timestamp"]==frame_id])
        
        label_points_homogeneous = np.concatenate([points_with_label[:, :3], 
                                                     np.ones((points_with_label.shape[0], 1))], 
                                                     axis=1)
        label_points_original_system = (label_points_homogeneous @ np.linalg.inv(lidar2world[0]).T)
        label_points_original_system[:,3] = points_with_label[:,3] 
        label_points_original_system= np.hstack((label_points_original_system,points_with_label[:,4].reshape(-1, 1))) 
        
        pano = LiDAR_2_Pano_KITTI(label_points_original_system, H, W, intrinsics)
        frame_name = Path(lidar_path).stem + ".npy"
        np.save(out_dir / frame_name, pano)

    # # Process images and save to 'image' directory
    # for img_path in tqdm(image_path, desc="Processing Images"):
    #     image_np = cv2.imread(img_path)
    #     if image_np is not None:
    #         frame_name = Path(img_path).stem + ".png"
    #         cv2.imwrite(str(image_out_dir / frame_name), image_np)
    #     else:
    #         print(f"Warning: Could not read image at {img_path}")

    # # Process semantic maps and save to 'semantic_map' directory
    # for img_path in tqdm(semantic_path, desc="Processing Semantic Maps"):
    #     image_np = cv2.imread(img_path)
    #     if image_np is not None:
    #         frame_name = Path(img_path).stem + ".png"
    #         cv2.imwrite(str(semantic_out_dir / frame_name), image_np)
    #     else:
    #         print(f"Warning: Could not read semantic map at {img_path}")

            
def create_kitti_rangeview(frame_start, frame_end, semantic_file_static, semantic_file_dynamic):
    data_root = Path(__file__).parent.parent
    kitti_360_root = data_root / "kitti360" / "KITTI-360"
    kitti_360_parent_dir = kitti_360_root.parent
    out_dir = kitti_360_parent_dir / "train"
    sequence_name = "2013_05_28_drive_0000"

    H = 66
    W = 1030
    intrinsics = (2.0, 26.9)  # fov_up, fov

    frame_ids = list(range(frame_start, frame_end + 1))

    lidar_dir = (
        kitti_360_root
        / "data_3d_raw"
        / f"{sequence_name}_sync"
        / "velodyne_points"
        / "data"
    )

    # img_dir = (
    #     kitti_360_root
    #     / "data_2d_raw"
    #     / f"{sequence_name}_sync"
    #     / "image_00"
    #     / "data_rect"
    # )

    # semantic_map = (
    #     kitti_360_root
    #     / "data_2d_raw"
    #     / f"{sequence_name}_sync"
    #     / "semantic_rgb"
    # )

    lidar_paths = [
        os.path.join(lidar_dir, "%010d.bin" % frame_id) for frame_id in frame_ids
    ]

    # image_path = [ 
    #     os.path.join(img_dir, "%010d.png" % frame_id) for frame_id in frame_ids 
    #     ]

    # semantic_path = [
    #     os.path.join(semantic_map, "%010d.png" % frame_id) for frame_id in frame_ids
    # ]

    generate_train_data(
        H=H,
        W=W,
        intrinsics=intrinsics,
        lidar_paths=lidar_paths,
        # image_path = image_path,
        # semantic_path = semantic_path,
        out_dir=out_dir,
        points_dim=4,
        kitti_360_root = kitti_360_root,
        sequence_name = sequence_name,
        semantic_file_static=semantic_file_static, 
        semantic_file_dynamic=semantic_file_dynamic
    )


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    # Check dataset.
    if args.dataset == "kitti360":
        frame_start = args.sequence_id
        if args.sequence_id == "1538":
            frame_start = 1538
            frame_end = 1601
        elif args.sequence_id == "1728":
            frame_start = 1728
            frame_end = 1791
        elif args.sequence_id == "1908":
            frame_start = 1908
            frame_end = 1971
        elif args.sequence_id == "3353":
            frame_start = 3353
            frame_end = 3416
        elif args.sequence_id == "2350":
            frame_start = 2350
            frame_end = 2400
        elif args.sequence_id == "4950":
            frame_start = 4950
            frame_end = 5000
        elif args.sequence_id == "8120":
            frame_start = 8120
            frame_end = 8170
        elif args.sequence_id == "10200":
            frame_start = 10200
            frame_end = 10250
        elif args.sequence_id == "10750":
            frame_start = 10750
            frame_end = 10800
        elif args.sequence_id == "11400":
            frame_start = 11400
            frame_end = 11450
        else:
            raise ValueError(f"Invalid sequence id: {sequence_id}")
        semantic_file_static = args.static_semantic_data_path
        semantic_file_dynamic = args.dynamic_semantic_data_path
        print(f"Generate rangeview from {frame_start} to {frame_end} ...")
        create_kitti_rangeview(frame_start, frame_end, semantic_file_static, semantic_file_dynamic)


if __name__ == "__main__":
    main()
