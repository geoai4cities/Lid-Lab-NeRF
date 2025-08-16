import os 
import numpy as np

def lidar_to_pano_with_intensities_and_labels(
    local_points_with_intensities_labels: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    lidar_K: int,
    max_depth=80,
):
    """
    Convert lidar frame to pano frame with intensities and semantic labels.
    Lidar points are in local coordinates.

    Args:
        local_points_with_intensities_labels: (N, 5), float32, in lidar frame, with intensities and semantic labels.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        max_depth: max depth in meters.

    Return:
        pano: (H, W), float32, depth values.
        intensities: (H, W), float32, intensity values.
        semantic_labels: (H, W), float32, semantic labels.
    """
    # Unpack points, intensities, and labels.
    local_points = local_points_with_intensities_labels[:, :3]
    local_point_intensities = local_points_with_intensities_labels[:, 3]
    local_point_labels = local_points_with_intensities_labels[:, 4]
    fov_up, fov = lidar_K
    fov_down = fov - fov_up

    # Compute distances to lidar center.
    dists = np.linalg.norm(local_points, axis=1)

    # Initialize output arrays.
    pano = np.zeros((lidar_H, lidar_W), dtype=np.float32)
    intensities = np.zeros((lidar_H, lidar_W), dtype=np.float32)
    semantic_labels = np.zeros((lidar_H, lidar_W), dtype=np.float32)

    for local_point, dist, intensity, label in zip(
        local_points, dists, local_point_intensities, local_point_labels
    ):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = local_point
        beta = np.pi - np.arctan2(y, x)
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
        c = int(round(beta / (2 * np.pi / lidar_W)))
        r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        # Set to min dist if not set, or if this distance is closer.
        if pano[r, c] == 0.0 or pano[r, c] > dist:
            pano[r, c] = dist
            intensities[r, c] = intensity
            semantic_labels[r, c] = label

    return pano, intensities, semantic_labels




gt_folder = "/root/docker_data/Lid-Lab-NeRF/data/kitti360/train"
sim_folder = "/root/docker_data/Lid-Lab-NeRF/log/kitti360_lidar4d_f4950_release/simulation/points"
import cv2
import matplotlib.pyplot as plt
gt_files = sorted(
    [f for f in os.listdir(gt_folder) if f.endswith(".npy")]
    )
    
sim_files = sorted(
    [f for f in os.listdir(sim_folder) if f.endswith(".npy")],
    key=lambda x: int(x.split('_')[1].split('.')[0])
)
simulated_scans = []
for file in sim_files:
    sim_scan = np.load(sim_folder + "/"+file)
    x,y,sim_label_map = lidar_to_pano_with_intensities_and_labels(sim_scan, 66,1030, (2,26.9))
    # print(np.unique(sim_label_map))
    simulated_scans.append(sim_label_map)
    plt.imshow(sim_label_map)
    break
ground_truth_scans = []
for file in gt_files:
    gt_label_map = np.load(gt_folder + "/"+file)[:,:,3]
    ground_truth_scans.append(gt_label_map)
    break


# Label mapping: sim -> gt
label_mapping = {
    0: 0,
    1: 7,
    2: 8,
    3: 11,
    4: 21,
    5: 26,
    6: 100
}

# Prepare IoU accumulators
class_ids = list(label_mapping.keys())  # sim labels
iou_sums = {cls: 0.0 for cls in class_ids}
counts = {cls: 0 for cls in class_ids}

# Iterate through pairs
for sim_map, gt_map in zip(simulated_scans, ground_truth_scans):
    for sim_label, gt_label in label_mapping.items():
        sim_mask = (sim_map == sim_label)
        gt_mask = (gt_map == gt_label)

        intersection = np.logical_and(sim_mask, gt_mask).sum()
        union = np.logical_or(sim_mask, gt_mask).sum()

        if union > 0:  # avoid division by zero
            iou = intersection / union
            iou_sums[sim_label] += iou
            counts[sim_label] += 1

# Compute mIoU per class
miou_per_class = {
    cls: (iou_sums[cls] / counts[cls]) if counts[cls] > 0 else 0.0
    for cls in class_ids
}

# Print results
print("Per-class mIoU:")
for cls in class_ids:
    print(f"Class {cls} (GT {label_mapping[cls]}): {miou_per_class[cls]:.4f}")

# Overall mean of mIoU values
overall_miou = np.mean(list(miou_per_class.values()))
print(f"Overall mIoU: {overall_miou:.4f}")


    
