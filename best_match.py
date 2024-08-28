import os
import numpy as np
import torch
import open3d as o3d
from gedi import GeDi

def compute_registration(src_path, target_path, model, voxel_size, patches_per_pair):
    # Load point clouds
    pcd0 = o3d.io.read_point_cloud(src_path)
    pcd1 = o3d.io.read_point_cloud(target_path)

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

def main(input_dir, target_path):
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
    registration_result = {}

    # Read source files in the folder
    sources_path = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.ply')]

    # For every source, compute the registration using GeDi and save the result
    for src_path in sources_path:
        registration_result[src_path] = compute_registration(src_path, target_path, model=gedi, voxel_size=voxel_size, patches_per_pair=patches_per_pair)

    # Evaluate which alignment is the best according to the fitness or rmse
    sorted_by_fitness = sorted(registration_result.items(), key=lambda x: x[1].fitness, reverse=True)
    sorted_by_rmse = sorted(registration_result.items(), key=lambda x: x[1].inlier_rmse, reverse=False)

    # Save the results
    with open('result.txt', 'a') as f:
        f.write("################################################\n")
        f.write(f"Target: {target_path}\n")
        
        f.write("\nSorted by RMSE:\n")
        for item in sorted_by_rmse:
            f.write(f"{item[0]}: fitness={item[1].fitness}, rmse={item[1].inlier_rmse}\n")
        
        f.write("\nSorted by Fitness:\n")
        for item in sorted_by_fitness:
            f.write(f"{item[0]}: fitness={item[1].fitness}, rmse={item[1].inlier_rmse}\n")

    print("Results saved in result.txt")
if __name__ == "__main__":
    for file in os.listdir("data/10000"):
        target = os.path.join("data/10000", file)
        input_dir = "data/10000_noisy_hard"
        main(input_dir, target)