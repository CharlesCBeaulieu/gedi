import open3d as o3d
import numpy as np
import os
import numpy as np
from gedi import GeDi
import torch
import matplotlib.pyplot as plt
import copy
import time
import json
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import copy


class ScanIdentification2:
    def __init__(
        self,
        scan_folder,
        scan_desc_folder,
        scan_eig_folder,
        cad_folder,
        cad_desc_folder,
        cad_eig_folder,
        metadata_path,
    ):
        self.scan_folder = scan_folder
        self.scan_desc_folder = scan_desc_folder
        self.scan_eig_folder = scan_eig_folder
        self.cad_folder = cad_folder
        self.cad_desc_folder = cad_desc_folder
        self.cad_eig_folder = cad_eig_folder
        self.metadata_path = metadata_path

    @staticmethod
    def compute_descriptors(pcd_path: str, model: GeDi):
        voxel_size = 0.01
        patches_per_pair = 5000

        # Load ply and process if needed
        pcd = o3d.io.read_point_cloud(pcd_path)

        # Downsample and estimate normals
        inds = np.random.choice(np.asarray(pcd.points).shape[0], patches_per_pair, replace=False)
        pts = torch.tensor(np.asarray(pcd.points)[inds]).float()
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd.estimate_normals()
        _pcd = torch.tensor(np.asarray(pcd.points)).float()
        pcd_desc = model.compute(pts=pts, pcd=_pcd)

        return pcd_desc

    @staticmethod
    def load_model():
        # model config
        config = {
            "dim": 32,  # descriptor output dimension
            "samples_per_batch": 500,  # batches to process the data on GPU
            "samples_per_patch_lrf": 4000,  # num. of point to process with LRF
            "samples_per_patch_out": 512,  # num. of points to sample for pointnet++
            "r_lrf": 0.5,  # LRF radius
            "fchkpt_gedi_net": "data/chkpts/3dmatch/chkpt.tar",  # path to checkpoint
        }
        # Load GeDi with config
        gedi = GeDi(config)

        return gedi

    @staticmethod
    def preprocess_scan(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) - np.mean(np.asarray(pcd.points), axis=0))
        return pcd

    @staticmethod
    def preprocess_cad(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        # Normalize and center
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) / 1000)
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) - np.mean(np.asarray(pcd.points), axis=0))
        return pcd

    @staticmethod
    def eig_similarity(scan_eig, cad_eig, penalty=True):
        # Compute similarity between scan and cad eigenvalues by % difference and sum the %
        # Add a penalty for compensating eigenvalues
        diff = np.abs(scan_eig - cad_eig) / scan_eig
        penalty = np.std(scan_eig) if penalty else 0
        return np.sum(diff) + penalty

    def compute_and_save_desc(self, gedi):
        total_time_scan = 0
        total_time_cad = 0

        # Compute and save descriptors for 5 scans
        for scan in tqdm(os.listdir(self.scan_folder)[:5], desc="Computing scan descriptors"):
            start = time.time()
            scan_path = os.path.join(self.scan_folder, scan)
            scan_desc = ScanIdentification2.compute_descriptors(scan_path, gedi)
            scan_desc_path = os.path.join(self.scan_desc_folder, scan.split(".")[0])
            np.save(scan_desc_path, scan_desc)
            end = time.time()
            t = end - start
            total_time_scan += t

        # Compute and save descriptors for 5 cads
        for cad in tqdm(os.listdir(self.cad_folder)[:5], desc="Computing cad descriptors"):
            start = time.time()
            cad_path = os.path.join(self.cad_folder, cad)
            cad_desc = ScanIdentification2.compute_descriptors(cad_path, gedi)
            cad_desc_path = os.path.join(self.cad_desc_folder, cad.split(".")[0])
            np.save(cad_desc_path, cad_desc)
            end = time.time()
            t = end - start
            total_time_cad += t

        print("Time taken for computing of the dest + IO operations for computing the descriptors")
        print("------------------------------------------------------")
        print(f"Total time for scan descriptors: {total_time_scan}")
        print(f"Total time for cad descriptors: {total_time_cad}")
        print(f"Total time: {total_time_scan + total_time_cad}")
        print(f"Average time per scan: {total_time_scan / 5}")
        print(f"Average time per cad: {total_time_cad / 5}")

    def compute_and_save_eig(self):
        start = time.time()
        print("Computing and saving eigenvalues for 5 scans and 5 cads folder")

        for scan in tqdm(os.listdir(self.scan_folder)[:5], desc="Computing scan eigenvalues"):
            scan_path = os.path.join(self.scan_folder, scan)
            pcd = o3d.io.read_point_cloud(scan_path)
            pcd = ScanIdentification2.preprocess_scan(pcd)
            eig = np.linalg.eig(np.cov(np.asarray(pcd.points).T))[0]
            np.save(os.path.join(self.scan_eig_folder, scan.split(".")[0]), eig)

        for cad in tqdm(os.listdir(self.cad_folder)[:5], desc="Computing cad eigenvalues"):
            cad_path = os.path.join(self.cad_folder, cad)
            pcd = o3d.io.read_point_cloud(cad_path)
            pcd = ScanIdentification2.preprocess_cad(pcd)
            eig = np.linalg.eig(np.cov(np.asarray(pcd.points).T))[0]
            np.save(os.path.join(self.cad_eig_folder, cad.split(".")[0]), eig)

        end = time.time()
        print("Done !")
        print(f"Time taken for computing and saving eigenvalues: {end - start:.2f} seconds")

    def coarse_filtering(self, scan_path):
        cad_score = {}
        cad_eig_dict = {}
        start = time.time()

        # Load scan eig and sort them
        scan_eig = np.load(os.path.join(self.scan_eig_folder, os.path.basename(scan_path).split(".")[0] + ".npy"))
        scan_eig = np.sort(scan_eig)[::-1]

        print("\033[92m✅ ----------------- Coarse Filtering ----------------- ✅\033[0m")
        print("\n")
        print(f"Target scan : {os.path.basename(scan_path)}")
        print(f"Target scan eigenvalues : {', '.join(f'{val:.4f}' for val in scan_eig[:3])}")

        # Compute similarity between scan and cads
        for cad in tqdm(os.listdir(self.cad_eig_folder), desc="Processing cads for coarse filtering"):
            cad_num = cad.split(".")[0]
            # load cad eig and sort them
            cad_eig_val = np.load(os.path.join(self.cad_eig_folder, cad))
            cad_eig_val = np.sort(cad_eig_val)[::-1]

            # similarity between scan and cad sorted eigenvalues
            similarity = ScanIdentification2.eig_similarity(scan_eig, cad_eig_val, penalty=True)
            cad_score[cad_num] = similarity
            cad_eig_dict[cad_num] = cad_eig_val

        sorted_cad_score = dict(sorted(cad_score.items(), key=lambda item: item[1]))

        end = time.time()
        print(f"⏳ Took : {end - start:.2f} seconds")

        print(f"{'| Index':<8}| {'Cad':<10}| {'Score (lower is better)':<25}| {'Eigenvalues':<30}|")
        print("-" * 80)  # Adjust width if necessary
        for idx, (cad, score) in enumerate(sorted_cad_score.items(), start=1):
            eig_values = ", ".join(f"{val:.4f}" for val in cad_eig_dict[cad][:3])  # Display first 3 eigenvalues
            print(f"| {idx:<6}| {cad:<10}| {score:<10.4f}               | {eig_values:<30}|")

        print("\n")
        return sorted_cad_score

    def fine_filtering(self, scan_path, candidate_cads):

        print("\033[92m✅ ----------------- Fine Filtering ----------------- ✅\033[0m")
        start = time.time()

        # Load the scan descriptors and scan point cloud.
        scan_basename = os.path.basename(scan_path).split(".")[0]
        scan_desc = np.load(os.path.join(self.scan_desc_folder, scan_basename + ".npy"))
        scan_pts = o3d.io.read_point_cloud(scan_path).points
        pcd0_dsdv = o3d.pipelines.registration.Feature()
        pcd0_dsdv.data = scan_desc.T

        registration_result = {}

        # Create an output folder for the combined point clouds.
        output_folder = "gedi_data/registration_results/gedi_fine_out"
        os.makedirs(output_folder, exist_ok=True)

        for cad in tqdm(candidate_cads, desc="Processing cads for fine filtering"):
            # Load the CAD descriptors and CAD point cloud.
            cad_desc = np.load(os.path.join(self.cad_desc_folder, cad + ".npy"))
            cad_pts = o3d.io.read_point_cloud(os.path.join(self.cad_folder, cad + ".ply")).points

            # Prepare features for registration.
            pcd1_dsdv = o3d.pipelines.registration.Feature()
            pcd1_dsdv.data = cad_desc.T

            _pcd0 = o3d.geometry.PointCloud()
            _pcd0.points = o3d.utility.Vector3dVector(scan_pts)
            _pcd1 = o3d.geometry.PointCloud()
            _pcd1.points = o3d.utility.Vector3dVector(cad_pts)

            if _pcd0.is_empty() or _pcd1.is_empty():
                raise ValueError("Empty point scan or cad")

            # Perform RANSAC-based registration.
            est_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                _pcd0,
                _pcd1,
                pcd0_dsdv,
                pcd1_dsdv,
                mutual_filter=True,
                max_correspondence_distance=0.02,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.02),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000),
            )

            registration_result[cad] = {
                "fitness": est_result.fitness,
                "inlier_rmse": est_result.inlier_rmse,
                "transformation": est_result.transformation,
            }

            # --- Create combined colored point clouds for visualization ---
            # Use the original point clouds directly (no copy).
            pcd_scan = _pcd0  # Original scan point cloud
            pcd_cad = _pcd1  # Original candidate CAD point cloud

            # Apply the estimated transformation to the scan point cloud.
            pcd_scan_copy = copy.deepcopy(pcd_scan)
            pcd_scan_copy.transform(est_result.transformation)
            # Color the transformed scan in red and the CAD in blue.
            pcd_scan_copy.paint_uniform_color([1, 0, 0])  # red
            pcd_cad_copy = copy.deepcopy(pcd_cad)
            pcd_cad_copy.paint_uniform_color([0, 0, 1])  # blue
            # Combine the two point clouds.
            combined_pcd = pcd_scan_copy + pcd_cad_copy
            # Define output filename.
            output_filename = os.path.join(output_folder, f"{cad}_aligned.ply")
            # Save the combined point cloud.
            o3d.io.write_point_cloud(output_filename, combined_pcd)
            print(f"Saved combined point cloud for CAD {cad} to {output_filename}")

        end = time.time()
        print(f"⏳ Took : {end - start:.2f} seconds")

        for cad, result in registration_result.items():
            print(f"Cad {cad} : fitness = {result['fitness']}, inlier_rmse = {result['inlier_rmse']}")

        return registration_result


if __name__ == "__main__":
    # Initialize the ScanIdentification2 instance with appropriate folders and metadata path
    identification = ScanIdentification2(
        scan_folder="gedi_data/gen_scan",
        scan_desc_folder="gedi_data/gen_scan_desc",
        scan_eig_folder="gedi_data/gen_scan_eig",
        cad_folder="gedi_data/gt_cad_scaled_0001",
        cad_desc_folder="gedi_data/gt_cad_desc",
        cad_eig_folder="gedi_data/gt_cad_eig",
        metadata_path="gedi_data/metadata.json",
    )
    gedi = ScanIdentification2.load_model()
    cum_error_threshold = 5

    # only need to run this line if the descriptors are not already computed
    identification.compute_and_save_desc(gedi)

    # Only need to run this line if the eigenvalues are not already computed
    # Compute eigenvalues for scans and cads
    identification.compute_and_save_eig()

    # Process coarse fitering
    scan = "763638.ply"
    scan_path = os.path.join(identification.scan_folder, scan)
    cad_score = identification.coarse_filtering(scan_path)

    # Process fine filtering
    candidate_cads = [cad for cad, score in cad_score.items() if score < cum_error_threshold]
    candidate_cads = candidate_cads[:5]
    identification.fine_filtering(scan_path, candidate_cads)
