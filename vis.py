import os
# Enable CPU rendering mode if necessary
os.environ["OPEN3D_CPU_RENDERING"] = "1"

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

# Load point cloud from a text file (assumes one x y z per line)
points = np.loadtxt("/root/docker_data/MSthesis/log/kitti360_lidar4d_f4950_release/simulation/points/lidar4d_0046.txt")[:,:3]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Set up offscreen renderer with desired resolution
width, height = 800, 600
renderer = rendering.OffscreenRenderer(width, height)

# Create a simple material for the point cloud
material = rendering.MaterialRecord()
material.shader = "defaultUnlit"  # or choose another shader as needed

# Add the point cloud geometry to the scene
renderer.scene.add_geometry("point_cloud", pcd, material)

# Set up the camera view:
# Compute the bounding box center and extent for a reasonable camera position.
bbox = pcd.get_axis_aligned_bounding_box()
center = bbox.get_center()
extent = bbox.get_extent()
# Position the camera at a distance based on the point cloud size.
eye = [center[0] + extent[0]*2, center[1] + extent[1]*2, center[2] + extent[2]*2]
up = [0, 1, 0]
renderer.scene.camera.look_at(center, eye, up)

# Render the scene to an image
image = renderer.render_to_image()

# Save the rendered image to a PNG file
o3d.io.write_image("output.png", image)
