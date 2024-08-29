import os
import numpy as np
import torch
import open3d as o3d
from src.gedi import GeDi

# Create the demo_output directory if it doesn't exist
output_dir = "demo_output0"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configuration for GeDi
config = {
    'dim': 32,  # Descriptor output dimension
    'samples_per_batch': 500,  # Batches to process the data on GPU
    'samples_per_patch_lrf': 4000,  # Num. of points to process with LRF
    'samples_per_patch_out': 512,  # Num. of points to sample for PointNet++
    'r_lrf': .5,  # LRF radius
    'fchkpt_gedi_net': 'data/chkpts/3dmatch/chkpt.tar'  # Path to checkpoint
}

voxel_size = .01
patches_per_pair = 5000

# Initialize GeDi
gedi = GeDi(config=config)

def normalize_point_cloud(pcd):
    # Convert to numpy array
    points = np.asarray(pcd.points)
    
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Center the point cloud
    points -= centroid
    
    # Compute scale
    scale = np.max(np.linalg.norm(points, axis=1))
    
    # Normalize to fit within unit sphere
    if scale != 0:
        points /= scale
    
    # Set the points back to the point cloud
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd

# Load point clouds
pcd0 = o3d.io.read_point_cloud("data/sample_10000/src_10000.ply")
pcd1 = o3d.io.read_point_cloud("data/sample_10000/tgt_10000.ply")

# Normalize point clouds
pcd0 = normalize_point_cloud(pcd0)
pcd1 = normalize_point_cloud(pcd1)

# Save the normalized point clouds
o3d.io.write_point_cloud(os.path.join(output_dir, "pcd0.ply"), pcd0)
o3d.io.write_point_cloud(os.path.join(output_dir, "pcd1.ply"), pcd1)

# Color the point clouds for visualization
pcd0.paint_uniform_color([1, 0.706, 0])
pcd1.paint_uniform_color([0, 0.651, 0.929])

# Estimate normals (only for visualization)
pcd0.estimate_normals()
pcd1.estimate_normals()

o3d.visualization.draw_geometries([pcd0, pcd1])

# Randomly sample points from the point cloud
inds0 = np.random.choice(np.asarray(pcd0.points).shape[0], patches_per_pair, replace=False)
inds1 = np.random.choice(np.asarray(pcd1.points).shape[0], patches_per_pair, replace=False)

pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float()
pts1 = torch.tensor(np.asarray(pcd1.points)[inds1]).float()

# Apply voxelization to the point cloud
pcd0 = pcd0.voxel_down_sample(voxel_size)
pcd1 = pcd1.voxel_down_sample(voxel_size)

_pcd0 = torch.tensor(np.asarray(pcd0.points)).float()
_pcd1 = torch.tensor(np.asarray(pcd1.points)).float()

# Compute descriptors
pcd0_desc = gedi.compute(pts=pts0, pcd=_pcd0)
pcd1_desc = gedi.compute(pts=pts1, pcd=_pcd1)

# Prepare format for Open3D RANSAC
pcd0_dsdv = o3d.pipelines.registration.Feature()
pcd1_dsdv = o3d.pipelines.registration.Feature()

pcd0_dsdv.data = pcd0_desc.T
pcd1_dsdv.data = pcd1_desc.T

_pcd0 = o3d.geometry.PointCloud()
_pcd0.points = o3d.utility.Vector3dVector(pts0)
_pcd1 = o3d.geometry.PointCloud()
_pcd1.points = o3d.utility.Vector3dVector(pts1)

# Apply RANSAC
est_result01 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    _pcd0,
    _pcd1,
    pcd0_dsdv,
    pcd1_dsdv,
    mutual_filter=True,
    max_correspondence_distance=.02,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=3,
    checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
              o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.02)],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000)
)

# Save the transformation matrix
transformation_path = os.path.join(output_dir, "transformation.txt")
np.savetxt(transformation_path, est_result01.transformation)

# Apply estimated transformation
pcd0.transform(est_result01.transformation)
o3d.visualization.draw_geometries([pcd0, pcd1])

print(f"Point clouds and transformation matrix saved in the {output_dir} folder.")