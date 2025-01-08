import open3d as o3d
import numpy as np
import os
import numpy as np
import open3d as o3d
from gedi import GeDi
import torch
import matplotlib.pyplot as plt
import copy


# The goal of this script is to identify the scans that are present in the data pool
# first step is to use a coarse method to filter most of the scans and then use 
# a more fine method to identify the scans that are present in the data pool

class ScanIdentification:
    def __init__(self, scan_path, cad_path, output_path):
        self.scan_path = scan_path
        self.cad_path = cad_path
        self.output_path = output_path
    
    @staticmethod
    def load_scan_and_process(scan_path : str) -> o3d.geometry.PointCloud:
        pcd = o3d.io.read_point_cloud(scan_path)
        
        # Scale the scan
        scan_points = np.asarray(pcd.points)
        scaled_scan_points = scan_points * 1000 
        
        # Center the scan
        centroid = np.mean(scaled_scan_points, axis=0)
        centered_scan_points = scaled_scan_points - centroid 
        
        pcd.points = o3d.utility.Vector3dVector(centered_scan_points)
        return pcd


    @staticmethod
    def coarse_identification(scan_pcd : o3d.geometry.PointCloud, cads_folder : str) -> dict:
        """
        This function will compare the eigenvalues of the scan to the eigenvalues of the cads
        and return the difference between the eigenvalues. The smallest difference will be the
        most likely candidate for the scan.
        """
        # step 1: load the scan and compute the eigenvalues
        # TODO this step could be better done with more features than just the eigenvalues
        # TODO for example, could use pointnet to extract features from the cads... 
        
        scan_eigs = np.linalg.eigvals(np.cov(np.asarray(scan_pcd.points).T))
        sorted_scan_eigs = np.sort(scan_eigs)[::-1] # sort the eigenvalues, biggest first
        
        results = {}
        for cad_file in os.listdir(cads_folder):
            cad_num = cad_file.split(".")[0]
            # step 2: load the cad and compute the eigenvalues
            cad_pcd = o3d.io.read_point_cloud(os.path.join(cads_folder, cad_file))
            cad_eigs = np.linalg.eigvals(np.cov(np.asarray(cad_pcd.points).T))
            sorted_cad_eigs = np.sort(cad_eigs)[::-1] # sort the eigenvalues, biggest first
            
            # compare the eigenvalues (I use the norm of the difference but could use other metrics)
            # TODO: could use other metrics to compare the eigenvalues
            diff = np.linalg.norm(sorted_scan_eigs - sorted_cad_eigs)
            results[cad_num] = diff
            
        # step 3 : sort the results and return the dict
        results = dict(sorted(results.items(), key=lambda item: item[1]))
        return results
    
    
    @staticmethod
    def gedi_compute(source : o3d.geometry.PointCloud, target : o3d.geometry.PointCloud, output_file : str) -> dict:
        config = {'dim': 32,                                      # descriptor output dimension
            'samples_per_batch': 500,                             # batches to process the data on GPU
            'samples_per_patch_lrf': 4000,                        # num. of point to process with LRF
            'samples_per_patch_out': 512,                         # num. of points to sample for pointnet++
            'r_lrf': .5,                                          # LRF radius
            'fchkpt_gedi_net': 'data/chkpts/3dmatch/chkpt.tar'}   # path to checkpoint

        voxel_size = .01
        patches_per_pair = 5000

        # initialising class
        gedi = GeDi(config=config)

        # getting a pair of point clouds
        pcd0 = source
        pcd1 = target
        pcd0.points = o3d.utility.Vector3dVector(np.asarray(pcd0.points)/1000)
        pcd1.points = o3d.utility.Vector3dVector(np.asarray(pcd1.points)/1000)
        # pcd0 = o3d.io.read_point_cloud('data/assets/threed_match_7-scenes-redkitchen_cloud_bin_0.ply')
        # pcd1 = o3d.io.read_point_cloud('data/assets/threed_match_7-scenes-redkitchen_cloud_bin_5.ply')
        
        save_two_pointclouds_as_image(pcd0, pcd1, output_file)
        
        # estimating normals (only for visualisation)
        pcd0.estimate_normals()
        pcd1.estimate_normals()

        # randomly sampling some points from the point cloud
        inds0 = np.random.choice(np.asarray(pcd0.points).shape[0], patches_per_pair, replace=False)
        inds1 = np.random.choice(np.asarray(pcd1.points).shape[0], patches_per_pair, replace=False)

        pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float()
        pts1 = torch.tensor(np.asarray(pcd1.points)[inds1]).float()

        # applying voxelisation to the point cloud
        pcd0 = pcd0.voxel_down_sample(voxel_size)
        pcd1 = pcd1.voxel_down_sample(voxel_size)

        _pcd0 = torch.tensor(np.asarray(pcd0.points)).float()
        _pcd1 = torch.tensor(np.asarray(pcd1.points)).float()

        # computing descriptors
        pcd0_desc = gedi.compute(pts=pts0, pcd=_pcd0)
        pcd1_desc = gedi.compute(pts=pts1, pcd=_pcd1)
        
        # preparing format for open3d ransac
        pcd0_dsdv = o3d.pipelines.registration.Feature()
        pcd1_dsdv = o3d.pipelines.registration.Feature()

        pcd0_dsdv.data = pcd0_desc.T
        pcd1_dsdv.data = pcd1_desc.T

        _pcd0 = o3d.geometry.PointCloud()
        _pcd0.points = o3d.utility.Vector3dVector(pts0)
        _pcd1 = o3d.geometry.PointCloud()
        _pcd1.points = o3d.utility.Vector3dVector(pts1)

        # applying ransac
        est_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
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
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))
        
        return est_result
    
    
    @ staticmethod
    def fine_identification(scan : o3d.geometry.PointCloud, candidates : dict) -> dict:
        """
        args:
        - scan : o3d.geometry.PointCloud : the scan to identify
        - candidate : dict : the candidate to compare to the scan (the output of coarse_identification)
        """
        # try to align the scan with the candidate
        # the one with the best alignment is the most likely to be the cad we are looking for
        for idx, candidate in enumerate(candidates):
            scan_c = copy.deepcopy(scan)
            candidate_pcd = o3d.io.read_point_cloud(os.path.join(CADS_FOLDER, candidate + ".ply"))
            gedi_result = ScanIdentification.gedi_compute(scan_c, candidate_pcd, output_file=f"gedi_result{idx}.png")
            print(gedi_result)
            
        pass
    
def save_two_pointclouds_as_image(pcd1, pcd2, output_file="pointclouds_comparison.png"):
    """
    Visualizes two point clouds in different colors and saves the plot as an image.
    """
    # Convert to NumPy arrays
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    # Create the plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first point cloud in red
    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='red', s=1, label='Point Cloud 1')

    # Plot the second point cloud in blue
    ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='blue', s=1, label='Point Cloud 2')

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Comparison of Two Point Clouds')
    ax.legend()

    # Save the plot
    plt.savefig(output_file, dpi=300)
    print(f"Figure saved as '{output_file}'.")

    

if __name__ == "__main__":
    SCANS_FOLDER = "data/generated_SCAN_pool/pcd"
    CADS_FOLDER = "data/sbi_CAD"
    OUTPUT_FOLDER = "data/scan_identification"
    TopCandidateLimit = 10
    
    # take a scan for exemple
    scan = ScanIdentification.load_scan_and_process(scan_path="data/generated_SCAN_pool/pcd/771612.ply")
    coarse_results = ScanIdentification.coarse_identification(scan_pcd=scan, cads_folder=CADS_FOLDER)
    # coarse_results output example (CAD ply file name : difference between eigenvalues of the cad and the scan): 
    # {'785150.ply': 6431.304684870268, 
    #  '771731.ply': 9135.402320685202}
    
    # from the coarse results, choose the N best candidates and use the fine identification method
    candidates = list(coarse_results.keys())[:TopCandidateLimit]
    fine_results = ScanIdentification.fine_identification(scan=scan, candidates=candidates)
        
    
    
    