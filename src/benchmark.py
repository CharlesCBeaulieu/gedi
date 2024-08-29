import numpy as np 
import open3d as o3d
import torch
import os
from src.gedi import GeDi

def compute_registration(src_pcd, tgt_pcd, model, voxel_size, patches_per_pair):
    # Load point clouds
    pcd0 = src_pcd
    pcd1 = tgt_pcd

    # Color the point clouds for visualization
    pcd0.paint_uniform_color([1, 0.706, 0])
    pcd1.paint_uniform_color([0, 0.651, 0.929])

    # Estimate normals (only for visualization)
    pcd0.estimate_normals()
    pcd1.estimate_normals()

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
    pcd0_desc = model.compute(pts=pts0, pcd=_pcd0)
    pcd1_desc = model.compute(pts=pts1, pcd=_pcd1)

    # Prepare format for Open3D RANSAC
    pcd0_dsdv = o3d.pipelines.registration.Feature()
    pcd1_dsdv = o3d.pipelines.registration.Feature()

    pcd0_dsdv.data = pcd0_desc.T
    pcd1_dsdv.data = pcd1_desc.T

    _pcd0 = o3d.geometry.PointCloud()
    _pcd0.points = o3d.utility.Vector3dVector(pts0.numpy())
    _pcd1 = o3d.geometry.PointCloud()
    _pcd1.points = o3d.utility.Vector3dVector(pts1.numpy())

    # Apply RANSAC
    est_result01 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        _pcd0,
        _pcd1,
        pcd0_dsdv,
        pcd1_dsdv,
        mutual_filter=True,
        max_correspondence_distance=0.02,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.02)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000)
    )
    return est_result01

def load_config():
     # Configuration for GeDi
    config = {
        'dim': 32,  # Descriptor output dimension
        'samples_per_batch': 500,  # Batches to process the data on GPU
        'samples_per_patch_lrf': 4000,  # Num. of points to process with LRF
        'samples_per_patch_out': 512,  # Num. of points to sample for PointNet++
        'r_lrf': 0.5,  # LRF radius
        'fchkpt_gedi_net': 'data/chkpts/3dmatch/chkpt.tar'  # Path to checkpoint
    }

    voxel_size = 0.01
    patches_per_pair = 5000

    # Initialize GeDi
    gedi = GeDi(config=config)
    
    return gedi

def data_augmentation(pcd_path, rotation_range, translation_range, noise_std, num_outliers):
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # randomly rotate the point cloud
    pcd.rotate(np.random.uniform(*rotation_range), np.array([0, 0, 1]))
    
    # randomly translate the point cloud
    pcd.translate(np.random.uniform(*translation_range, 3))
    
    # add random noise to the point cloud
    noise = noise_std * np.random.randn(np.asarray(pcd.points).shape[0], 3)
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + noise)
    
    # add random outliers to the point cloud
    outliers = np.random.uniform(-1, 1, (num_outliers, 3))
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + outliers)
    
    return pcd
    

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

def benchmark(input_dir, target_path):

    # Load the GeDi model
    model = load_config()

    for source in os.listdir(input_dir):
        source = os.path.join(input_dir, source)
        source = data_augmentation(source, rotation_range=(-np.pi, np.pi), translation_range=(-0.1, 0.1), noise_std=0.01, num_outliers=100)
        result = compute_registration(source, target_path, model, voxel_size=0.01, patches_per_pair=5000)
        
    
    return est_result
    
    
    