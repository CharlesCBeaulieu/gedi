import os
import open3d as o3d
import numpy as np
from gedi import GeDi
import copy
import time
from tqdm import tqdm
import torch
import csv
import re
import pandas as pd


class ScanIdentification2:
    def __init__(self, base_path="gedi_data/working_data"):
        # Define the base path
        self.base_path = base_path

        # raw data folders
        self.scan_best_folder = self._create_directory("scan/scan_best/point_cloud")
        self.scan_median_folder = self._create_directory("scan/scan_median/point_cloud")
        self.cad_folder = self._create_directory("cad/point_cloud")

        # preprocess folders
        self.scan_best_preprocess_folder = self._create_directory("scan/scan_best/point_cloud/preprocess")
        self.scan_median_preprocess_folder = self._create_directory("scan/scan_median/point_cloud/preprocess")
        self.cad_preprocess_folder = self._create_directory("cad/point_cloud/preprocess")

        # descriptor and eigenvalue folders
        # scan best
        self.scan_best_desc_folder = self._create_directory("scan/scan_best/desc")
        self.scan_best_inds_folder = self._create_directory("scan/scan_best/inds")
        self.scan_best_eig_folder = self._create_directory("scan/scan_best/eigs")
        # scan median
        self.scan_median_desc_folder = self._create_directory("scan/scan_median/desc")
        self.scan_median_inds_folder = self._create_directory("scan/scan_median/inds")
        self.scan_median_eig_folder = self._create_directory("scan/scan_median/eigs")
        # cad
        self.cad_desc_folder = self._create_directory("cad/desc")
        self.cad_inds_folder = self._create_directory("cad/inds")
        self.cad_eig_folder = self._create_directory("cad/eigs")

    def __repr__(self):
        return f"ScanIdentification2(base_path={self.base_path})"

    def _create_directory(self, relative_path: str) -> str:
        """Construct and create a directory at base_path/relative_path."""
        full_path = os.path.join(self.base_path, relative_path)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def _get_base_name(self, file_name: str) -> str:
        """
        Example:
            "12432_9.npy" -> "12432"
        """
        base = os.path.splitext(file_name)[0]
        return base.split("_")[0]

    def _process_folder(self, input_folder, output_desc_folder, output_inds_folder, descriptor_type, model):
        """
        Process all .ply files in the input folder by computing descriptors using the provided model,
        and save the resulting descriptors and indices to the specified output folders.
        Skips files if descriptors already exist.

        Uses the _get_base_name function to rename files (only keeping the part before the first underscore).

        Returns:
            total_time (float): Total time taken to process the folder.
        """
        import time
        from tqdm import tqdm

        total_time = 0
        scan_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".ply")]
        for file in tqdm(scan_files, desc=f"Computing {descriptor_type} descriptors"):
            base_name = self._get_base_name(file)
            desc_path = os.path.join(output_desc_folder, f"{base_name}.npy")
            inds_path = os.path.join(output_inds_folder, f"{base_name}.npy")

            # Skip file if descriptors already computed
            if os.path.exists(desc_path) and os.path.exists(inds_path):
                print(f"Skipping {file} (already computed)")
                continue

            start = time.time()
            file_path = os.path.join(input_folder, file)
            desc, inds = self.compute_descriptors(file_path, model, patches_per_pair=3000)
            np.save(desc_path, desc)
            np.save(inds_path, inds)
            total_time += time.time() - start
        return total_time

    def preprocess_scan(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Center the scan point cloud."""
        points = np.asarray(pcd.points)
        original_centroid = np.mean(points, axis=0)
        centered_points = points - original_centroid
        pcd.points = o3d.utility.Vector3dVector(centered_points)
        return pcd

    def preprocess_cad(self, pcd: o3d.geometry.PointCloud, scale_factor: float) -> o3d.geometry.PointCloud:
        """Scale and center the CAD point cloud."""
        orig_points = np.asarray(pcd.points)
        scaled_points = orig_points * scale_factor
        pcd.points = o3d.utility.Vector3dVector(scaled_points)
        new_points = np.asarray(pcd.points)
        new_centroid = np.mean(new_points, axis=0)
        centered_points = new_points - new_centroid
        pcd.points = o3d.utility.Vector3dVector(centered_points)
        return pcd

    def preprocess_scans(self, scan_type="best"):
        """
        process scans, center them and save them in the 'preprocess' subfolder.
        scan_type: str, "best" or "median"
        """
        # select the right folders
        scan_folder = self.scan_best_folder if scan_type == "best" else self.scan_median_folder
        output_folder = self.scan_best_preprocess_folder if scan_type == "best" else self.scan_median_preprocess_folder

        os.makedirs(output_folder, exist_ok=True)

        # get all the files in the folder
        scan_files = [f for f in os.listdir(scan_folder) if f.lower().endswith(".ply")]

        for filename in tqdm(scan_files, desc=f"Prétraitement des scans ({scan_type})", unit="scan"):
            full_path = os.path.join(scan_folder, filename)
            pcd = o3d.io.read_point_cloud(full_path)
            preprocessed_pcd = self.preprocess_scan(pcd)

            # Get base name (only text before '_' and without extension)
            base_name = filename.split("_")[0]
            output_filename = base_name + ".ply"

            output_path = os.path.join(output_folder, output_filename)

            # Save the preprocessed point cloud only if it has more than 5000 points
            if len(pcd.points) > 5000:
                o3d.io.write_point_cloud(output_path, preprocessed_pcd)

    def preprocess_all_cads(self, scale_factor: float):
        """
        Process all CAD point clouds and save them in the 'preprocess' subfolder.
        scale_factor: float, scaling factor for the CADs.
        """
        # create the output folder
        output_folder = os.path.join(self.cad_folder, "preprocess")
        os.makedirs(output_folder, exist_ok=True)

        # get all the files in the folder
        cad_files = [f for f in os.listdir(self.cad_folder) if f.lower().endswith(".ply")]

        for filename in tqdm(cad_files, desc="Preprocessing CADs", unit="cad"):
            full_path = os.path.join(self.cad_folder, filename)
            pcd = o3d.io.read_point_cloud(full_path)
            preprocessed_pcd = self.preprocess_cad(pcd, scale_factor)
            output_path = os.path.join(output_folder, filename)
            o3d.io.write_point_cloud(output_path, preprocessed_pcd)

    def load_model(self):
        # Model configuration
        config = {
            "dim": 32,  # descriptor output dimension
            "samples_per_batch": 10,
            "samples_per_patch_lrf": 3000,
            "samples_per_patch_out": 512,
            "r_lrf": 0.5,
            "fchkpt_gedi_net": "data/chkpts/3dmatch/chkpt.tar",
        }
        gedi = GeDi(config)
        return gedi

    def compute_and_save_desc(self, model, scan_type="best"):
        """
        Compute and save descriptors for scans and CADs.
        model: GeDi, the model to use for descriptor computation.
        scan_type: str, "best" or "median"
        """
        if scan_type not in ["best", "median"]:
            raise ValueError("scan_type must be 'best' or 'median'.")

        # Sélection des dossiers selon le type de scan
        scan_preprocess_folder = self.scan_best_preprocess_folder if scan_type == "best" else self.scan_median_preprocess_folder
        scan_desc_folder = self.scan_best_desc_folder if scan_type == "best" else self.scan_median_desc_folder
        scan_inds_folder = self.scan_best_inds_folder if scan_type == "best" else self.scan_median_inds_folder

        total_time_scan = self._process_folder(
            input_folder=scan_preprocess_folder,
            output_desc_folder=scan_desc_folder,
            output_inds_folder=scan_inds_folder,
            descriptor_type=f"scan_{scan_type}",
            model=model,
        )

        # Vérifier si les fichiers CAD existent déjà avant de recalculer
        cad_files = [f for f in os.listdir(self.cad_preprocess_folder) if f.lower().endswith(".ply")]
        cad_already_computed = all(
            os.path.exists(os.path.join(self.cad_desc_folder, f"{os.path.splitext(f)[0]}.npy"))
            and os.path.exists(os.path.join(self.cad_inds_folder, f"{os.path.splitext(f)[0]}.npy"))
            for f in cad_files
        )

        total_time_cad = 0
        if not cad_already_computed:
            total_time_cad = self._process_folder(
                input_folder=self.cad_preprocess_folder,
                output_desc_folder=self.cad_desc_folder,
                output_inds_folder=self.cad_inds_folder,
                descriptor_type="cad",
                model=model,
            )
        else:
            print("Skipping CAD descriptor computation (already computed for all files)")

        print("Time taken for computing descriptors + IO operations:")
        print("------------------------------------------------------")
        print(f"Total time for scan_{scan_type} descriptors: {total_time_scan:.2f} seconds")
        print(f"Total time for CAD descriptors: {total_time_cad:.2f} seconds")
        print(f"Total time: {total_time_scan + total_time_cad:.2f} seconds")
        print(f"Average time per scan: {total_time_scan / max(1, len(os.listdir(scan_preprocess_folder))):.2f} seconds")
        if not cad_already_computed:
            print(f"Average time per CAD: {total_time_cad / max(1, len(cad_files)):.2f} seconds")

    def compute_and_save_eig(self, scan_type="best"):
        """
        Compute and save eigenvalues for the specified scan type ('best' or 'median').
        Also computes CAD eigenvalues if they have not been computed yet.

        Parameters:
        - scan_type (str): "best" or "median" to specify which scan set to process.
        """
        if scan_type not in ["best", "median"]:
            raise ValueError("scan_type must be 'best' or 'median'.")

        start = time.time()
        print(f"\033[93m🛠️ Computing and saving eigenvalues for {scan_type} scans... 🛠️\033[0m")

        # Select the correct folders based on scan_type
        scan_preprocess_folder = self.scan_best_preprocess_folder if scan_type == "best" else self.scan_median_preprocess_folder
        scan_eig_folder = self.scan_best_eig_folder if scan_type == "best" else self.scan_median_eig_folder

        # Process scan eigenvalues
        for scan in tqdm(os.listdir(scan_preprocess_folder), desc=f"Computing {scan_type} scan eigenvalues"):
            scan_path = os.path.join(scan_preprocess_folder, scan)
            pcd = o3d.io.read_point_cloud(scan_path)

            if len(pcd.points) == 0:
                print(f"Skipping {scan}: empty point cloud.")
                continue

            eig_values = np.linalg.eig(np.cov(np.asarray(pcd.points).T))[0]
            np.save(os.path.join(scan_eig_folder, scan.split(".")[0]), eig_values)

        # Check if CAD eigenvalues are already computed
        cad_files = [f for f in os.listdir(self.cad_preprocess_folder) if f.lower().endswith(".ply")]
        cad_already_computed = all(
            os.path.exists(os.path.join(self.cad_eig_folder, f"{os.path.splitext(f)[0]}.npy")) for f in cad_files
        )

        if not cad_already_computed:
            print("\033[93m🛠️ Computing CAD eigenvalues... 🛠️\033[0m")

            for cad in tqdm(cad_files, desc="Computing CAD eigenvalues"):
                cad_path = os.path.join(self.cad_preprocess_folder, cad)
                pcd = o3d.io.read_point_cloud(cad_path)

                if len(pcd.points) == 0:
                    print(f"Skipping {cad}: empty point cloud.")
                    continue

                eig_values = np.linalg.eig(np.cov(np.asarray(pcd.points).T))[0]
                np.save(os.path.join(self.cad_eig_folder, cad.split(".")[0]), eig_values)
        else:
            print("\033[92m✅ Skipping CAD eigenvalue computation (already computed for all files)\033[0m")

        end = time.time()
        print("\033[92m✅ Done!\033[0m")
        print(f"⏳ Time taken for computing and saving eigenvalues: {end - start:.2f} seconds")

    def coarse_filtering(self, scan_path):
        """
        Perform coarse filtering by comparing eigenvalue similarity between a scan and CADs,
        then display, save the results, and return a dictionary of statistics.

        Determines whether the scan is from the 'best' or 'median' folder and uses the corresponding eigenvalue folder.
        """

        cad_score = {}
        cad_eig_dict = {}
        start = time.time()

        # Extract target scan base name (only text before the dot)
        target_scan = os.path.basename(scan_path).split(".")[0]

        # Choose the appropriate scan eigenvalue folder based on the scan_path
        if "scan_best" in scan_path:
            scan_eig_folder = self.scan_best_eig_folder
        elif "scan_median" in scan_path:
            scan_eig_folder = self.scan_median_eig_folder
        else:
            raise ValueError("scan_path must contain 'scan_best' or 'scan_median'.")

        # Load scan eigenvalues and sort them.
        scan_eig = np.load(os.path.join(scan_eig_folder, target_scan + ".npy"))
        scan_eig = np.sort(scan_eig)[::-1]

        print("\033[93m🛠️ ----------------- Coarse Filtering / Eigenvalues Filtering (In Progress...) ----------------- 🛠️\033[0m")
        print(f"Target scan: {target_scan}")
        print(f"Target scan eigenvalues: {', '.join(f'{val:.4f}' for val in scan_eig[:3])}")

        # Process CADs (using global N for number of CAD files to process)
        for cad in tqdm(os.listdir(self.cad_eig_folder)[:N], desc="Processing CADs for coarse filtering"):
            cad_num = cad.split(".")[0]
            cad_eig_val = np.load(os.path.join(self.cad_eig_folder, cad))
            cad_eig_val = np.sort(cad_eig_val)[::-1]
            similarity = ScanIdentification2.eig_similarity(scan_eig, cad_eig_val, penalty=False, dim_weight=[1,1,1])
            cad_score[cad_num] = similarity
            cad_eig_dict[cad_num] = cad_eig_val

        sorted_cad_score = dict(sorted(cad_score.items(), key=lambda item: item[1]))
        end = time.time()

        # Determine the coarse rank and cum_error of the target scan (if present)
        target_coarse_rank = None
        target_cum_error = None
        for rank, (cad, score) in enumerate(sorted_cad_score.items(), start=1):
            if cad == target_scan:
                target_coarse_rank = rank
                target_cum_error = score
                break

        # Display results on console.
        print(f"⏳ Took: {end - start:.2f} seconds")
        header = f"| {'Rank':^6} | {'CAD':^10} | {'Score (lower is better)':^25} | {'Eigenvalues':^30} | {'Status':^8} |"
        print(header)
        print("-" * len(header))
        for idx, (cad, score) in enumerate(sorted_cad_score.items(), start=1):
            eig_values = ", ".join(f"{val:.4f}" for val in cad_eig_dict[cad][:3])
            status = ""
            if cad == target_scan:
                status = "✅" if idx == 1 else "⬅️"
            print(f"| {idx:^6} | {cad:^10} | {score:^25.4f} | {eig_values:^30} | {status:^8} |")
        print("-" * len(header))
        print("\033[92m✅ ----------------- Coarse Filtering / Eigenvalues Filtering (Done!) ----------------- ✅\033[0m")

        # Save results to CSV in a target-specific folder.
        results_folder = os.path.join(self.base_path, "results/csv")
        os.makedirs(results_folder, exist_ok=True)
        target_folder = os.path.join(results_folder, f"{target_scan}_exp")
        os.makedirs(target_folder, exist_ok=True)
        csv_file_path = os.path.join(target_folder, "coarse_filtering_results.csv")
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Rank", "CAD", "Score (lower is better)", "Eigenvalues", "Status"])
            for idx, (cad, score) in enumerate(sorted_cad_score.items(), start=1):
                eig_values = ", ".join(f"{val:.4f}" for val in cad_eig_dict[cad][:3])
                status = ""
                if cad == target_scan:
                    status = "✅" if idx == 1 else "⬅️"
                writer.writerow([idx, cad, f"{score:.4f}", eig_values, status])
        print(f"CSV file saved to: {csv_file_path}")

        # Prepare and return statistics.
        coarse_stats = {
            "target_scan": target_scan,
            "processing_time": end - start,
            "sorted_scores": sorted_cad_score,
            "target_coarse_rank": target_coarse_rank,
            "target_cum_error": target_cum_error,
        }
        return coarse_stats

    def fine_filtering(self, scan_path, candidate_cads):
        """
        Perform fine filtering (registration) for candidate CADs using RANSAC.
        For each candidate, compute an initial alignment using RANSAC and save results.
        Returns a tuple: (registration_result, fine_stats)

        The method automatically selects the correct scan descriptors and indices folders based on whether the scan
        belongs to the 'best' or 'median' preprocessed scans.
        """
        import csv
        import copy
        import torch
        from tqdm import tqdm

        print("\033[93m🛠️ ----------------- Fine Filtering / Registration (In progress...) ----------------- 🛠️\033[0m")
        start = time.time()

        # Extract scan base name.
        scan_basename = os.path.basename(scan_path).split(".")[0]
        target_scan = scan_basename

        # Load the scan point cloud.
        pcd0 = o3d.io.read_point_cloud(scan_path)

        # Select appropriate folders for scan descriptors and indices.
        if "scan_best" in scan_path:
            scan_inds_folder = self.scan_best_inds_folder
            scan_desc_folder = self.scan_best_desc_folder
        elif "scan_median" in scan_path:
            scan_inds_folder = self.scan_median_inds_folder
            scan_desc_folder = self.scan_median_desc_folder
        else:
            raise ValueError("scan_path must contain 'scan_best' or 'scan_median'.")

        # Load precomputed indices and sample points.
        inds0 = np.load(os.path.join(scan_inds_folder, scan_basename + ".npy"))
        pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float()

        # Voxel downsample the scan and build a new point cloud from sampled points.
        pcd0 = pcd0.voxel_down_sample(0.001)
        _pcd0 = o3d.geometry.PointCloud()
        _pcd0.points = o3d.utility.Vector3dVector(pts0)

        # Load precomputed scan descriptors.
        pcd0_desc = np.load(os.path.join(scan_desc_folder, scan_basename + ".npy"))
        pcd0_dsdv = o3d.pipelines.registration.Feature()
        pcd0_dsdv.data = pcd0_desc.T

        registration_result = {}

        # Create output folders (target-specific)
        results_folder = os.path.join(self.base_path, "results/csv")
        os.makedirs(results_folder, exist_ok=True)
        target_folder = os.path.join(results_folder, f"{target_scan}_exp")
        os.makedirs(target_folder, exist_ok=True)
        viz_folder = os.path.join(target_folder, "viz_results")
        os.makedirs(viz_folder, exist_ok=True)

        # Process candidate CADs (only the first N candidates).
        for idx, cad in enumerate(tqdm(candidate_cads, desc="Processing CADs for fine filtering")):
            # Load CAD point cloud.
            cad_path = os.path.join(self.cad_preprocess_folder, cad + ".ply")
            pcd1 = o3d.io.read_point_cloud(cad_path)

            # Load precomputed indices and sample points for the CAD.
            inds1 = np.load(os.path.join(self.cad_inds_folder, cad + ".npy"))
            pts1 = torch.tensor(np.asarray(pcd1.points)[inds1]).float()

            # Voxel downsample the CAD.
            pcd1 = pcd1.voxel_down_sample(0.001)

            # Build candidate CAD point cloud from sampled points.
            _pcd1 = o3d.geometry.PointCloud()
            _pcd1.points = o3d.utility.Vector3dVector(pts1)

            # Load precomputed CAD descriptors.
            pcd1_desc = np.load(os.path.join(self.cad_desc_folder, cad + ".npy"))
            pcd1_dsdv = o3d.pipelines.registration.Feature()
            pcd1_dsdv.data = pcd1_desc.T

            if _pcd0.is_empty() or _pcd1.is_empty():
                raise ValueError("Empty point cloud for scan or CAD.")

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

            ransac_inliers = len(est_result.correspondence_set) if hasattr(est_result, "correspondence_set") else "N/A"
            registration_result[cad] = {
                "ransac_fitness": est_result.fitness,
                "ransac_inlier_rmse": est_result.inlier_rmse,
                "ransac_transformation": est_result.transformation,
                "num_inliers": ransac_inliers,
            }

            # Save combined point cloud after RANSAC.
            pcd0_copy = copy.deepcopy(_pcd0)
            pcd0_copy.paint_uniform_color([0, 0.651, 0.929])  # Light Blue
            pcd0_copy.transform(est_result.transformation)
            pcd1.paint_uniform_color([1, 0.706, 0])  # Yellow
            combined_ransac = pcd0_copy + pcd1
            ransac_filename = os.path.join(viz_folder, f"{cad}_aligned_ransac.ply")
            o3d.io.write_point_cloud(ransac_filename, combined_ransac)

        end = time.time()
        print(f"⏳ Registration took: {end - start:.2f} seconds")

        # Process and sort registration results filtered by an inlier RMSE threshold.
        inlier_threshold = 0.02
        filtered_results = [
            (cad, data["ransac_fitness"], data["ransac_inlier_rmse"], data["num_inliers"])
            for cad, data in registration_result.items()
            if data["ransac_inlier_rmse"] <= inlier_threshold
        ]
        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

        print("\nSorted Registration Results (filtered by RANSAC inlier_rmse <= {:.2f}):".format(inlier_threshold))
        header = "| {0:^5} | {1:^15} | {2:^12} | {3:^15} | {4:^12} | {5:^8} |".format(
            "Rank", "CAD", "RANSAC Fit", "RANSAC RMSE", "Num Inliers", "Status"
        )
        print(header)
        print("-" * len(header))
        target_fine_rank = None
        best_fitness = None
        best_rmse = None
        for rank, (cad, r_fitness, r_rmse, num_inliers) in enumerate(sorted_results, start=1):
            status = "✅" if (cad == target_scan and rank == 1) else ("⬅️" if cad == target_scan else "")
            if cad == target_scan:
                target_fine_rank = rank
                best_fitness = r_fitness
                best_rmse = r_rmse
            print(
                "| {0:^5} | {1:^15} | {2:^12.4f} | {3:^15.4f} | {4:^12} | {5:^8} |".format(
                    rank, cad, r_fitness, r_rmse, num_inliers, status
                )
            )
        print("-" * len(header))
        print("\033[92m✅ ----------------- Fine Filtering (Done!) ----------------- ✅\033[0m")

        # Write fine filtering results to CSV.
        csv_file_path_fine = os.path.join(target_folder, "fine_filtering_results.csv")
        with open(csv_file_path_fine, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Rank", "CAD", "RANSAC Fitness", "RANSAC RMSE", "Num Inliers", "Status"])
            for rank, (cad, r_fitness, r_rmse, num_inliers) in enumerate(sorted_results, start=1):
                status = "✅" if (cad == target_scan and rank == 1) else ("⬅️" if cad == target_scan else "")
                writer.writerow([rank, cad, f"{r_fitness:.4f}", f"{r_rmse:.4f}", num_inliers, status])
        print(f"CSV file saved to: {csv_file_path_fine}")

        # Prepare fine filtering stats.
        fine_stats = {
            "target_scan": target_scan,
            "processing_time": end - start,
            "fine_rank": target_fine_rank,
            "best_fitness": best_fitness,
            "best_rmse": best_rmse,
            "nb_inliers_correspondence_set": num_inliers,
            "sorted_results": sorted_results,
        }
        return registration_result, fine_stats

    @staticmethod
    def eig_similarity(scan_eig, cad_eig, penalty=True, dim_weight=[1, 0.5, 0.25]):
        """
        Compute a similarity score between two scans based on their sorted eigenvalues.
        A perfect match (e.g. [1, 1, 1] vs [1, 1, 1]) results in a score of 0.
        Differences are computed as relative differences, weighted per dimension.
        The L2 norm of these weighted differences is used so that a high error in one
        dimension is not completely compensated by lower errors in others.
        
        Args:
            scan_eig (array-like): Eigenvalues from the scan.
            cad_eig (array-like): Eigenvalues from the CAD model.
            penalty (bool): Whether to add a penalty based on the standard deviation of scan_eig.
            dim_weight (list or array): Weights applied to the differences in each of the first three dimensions.
            
        Returns:
            float: A similarity score (lower means more similar).
        """
        # Ensure inputs are numpy arrays
        scan_eig = np.asarray(scan_eig)
        cad_eig = np.asarray(cad_eig)
        
        # Compute relative differences for each eigenvalue
        diff = np.abs(scan_eig - cad_eig) / scan_eig
        
        # Apply weights to the first three dimensions
        weighted_diff = diff[:3] * np.array(dim_weight)
        
        # Use L2 norm so that a high difference in any dimension has a stronger effect
        score = np.linalg.norm(weighted_diff, ord=2)
        
        # Optionally, add a penalty term (e.g. standard deviation of scan eigenvalues)
        penalty_val = np.std(scan_eig) if penalty else 0
        
        return score + penalty_val

    @staticmethod
    def compute_descriptors(pcd_path: str, model, voxel_size: float = 0.001, patches_per_pair: int = 5000):
        pcd = o3d.io.read_point_cloud(pcd_path)

        points = np.asarray(pcd.points)
        if points.shape[0] == 0:
            raise ValueError("The point cloud contains no points.")

        inds = np.random.choice(points.shape[0], patches_per_pair, replace=False)
        pts_sample = points[inds]
        pts_tensor = torch.tensor(pts_sample).float()
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.estimate_normals()
        down_points = np.asarray(pcd_down.points)
        pcd_down_tensor = torch.tensor(down_points).float()
        descriptor = model.compute(pts=pts_tensor, pcd=pcd_down_tensor)
        return (descriptor, inds)

    @staticmethod
    # Usefull script
    def whole_pipeline1(preprocess_scan=True, preprocess_cad=True, compute_eig=True, compute_desc=True):
        N = 314  # Maximum number of CAD files to process
        cum_error_threshold = 1  # Cumulative error threshold for coarse filtering

        # Load GEDI model.
        identification = ScanIdentification2(base_path="gedi_data/working_data")
        gedi = identification.load_model()

        if preprocess_scan:
            # preprocess scans
            identification.preprocess_scans(scan_type="best")
            identification.preprocess_scans(scan_type="median")

        # preprocess CADs
        if preprocess_cad:
            identification.preprocess_all_cads(scale_factor=0.001)

        # Pre-compute eigenvalues for best and median scans.
        if compute_eig:
            identification.compute_and_save_eig(scan_type="best")
            identification.compute_and_save_eig(scan_type="median")

        # compute descriptors for best and median scans
        if compute_desc:
            identification.compute_and_save_desc(gedi, scan_type="best")
            identification.compute_and_save_desc(gedi, scan_type="median")

        # Dictionaries to store stats for each processed scan per experiment.
        overall_stats_best = {}
        overall_stats_median = {}

        # Process scans for the 'best' experiment.
        scan_preprocess_folder_best = identification.scan_best_preprocess_folder
        scans_best = [file for file in os.listdir(scan_preprocess_folder_best) if file.lower().endswith(".ply")]

        # Process scans for the 'median' experiment.
        scan_preprocess_folder_median = identification.scan_median_preprocess_folder
        scans_median = [file for file in os.listdir(scan_preprocess_folder_median) if file.lower().endswith(".ply")]

        for scan in scans_best:
            scan_path = os.path.join(scan_preprocess_folder_best, scan)
            print(f"\nProcessing best scan: {scan_path}")
            coarse_stats = identification.coarse_filtering(scan_path)
            candidate_cads = [cad for cad, score in coarse_stats["sorted_scores"].items() if score < cum_error_threshold]
            _, fine_stats = identification.fine_filtering(scan_path, candidate_cads)
            overall_stats_best[os.path.basename(scan_path)] = {"coarse": coarse_stats, "fine": fine_stats}

        for scan in scans_median:
            scan_path = os.path.join(scan_preprocess_folder_median, scan)
            print(f"\nProcessing median scan: {scan_path}")
            coarse_stats = identification.coarse_filtering(scan_path)
            candidate_cads = [cad for cad, score in coarse_stats["sorted_scores"].items() if score < cum_error_threshold]
            _, fine_stats = identification.fine_filtering(scan_path, candidate_cads)
            overall_stats_median[os.path.basename(scan_path)] = {"coarse": coarse_stats, "fine": fine_stats}

        # Build DataFrame for the 'best' experiment.
        data_best = []
        for scan, stats in overall_stats_best.items():
            coarse = stats["coarse"]
            fine = stats["fine"]
            data_best.append(
                {
                    "scan": scan,
                    "coarse_processing_time": coarse.get("processing_time", None),
                    "coarse_rank": coarse.get("target_coarse_rank", None),
                    "coarse_cum_error": coarse.get("target_cum_error", None),
                    "fine_processing_time": fine.get("processing_time", None),
                    "fine_rank": fine.get("fine_rank", None),
                    "fine_fitness": fine.get("best_fitness", None),
                    "fine_corr_rmse": fine.get("best_rmse", None),
                    "fine_nb_inliers_correspondence_set": fine.get("nb_inliers_correspondence_set", None),
                }
            )

        df_best = pd.DataFrame(data_best)
        print("Overall Stats DataFrame (Best):")
        print(df_best)
        df_best.to_csv("gedi_data/working_data/stats_results/overall_stats_best.csv", index=False)

        # Build DataFrame for the 'median' experiment.
        data_median = []
        for scan, stats in overall_stats_median.items():
            coarse = stats["coarse"]
            fine = stats["fine"]
            data_median.append(
                {
                    "scan": scan,
                    "coarse_processing_time": coarse.get("processing_time", None),
                    "coarse_rank": coarse.get("target_coarse_rank", None),
                    "coarse_cum_error": coarse.get("target_cum_error", None),
                    "fine_processing_time": fine.get("processing_time", None),
                    "fine_rank": fine.get("fine_rank", None),
                    "fine_fitness": fine.get("best_fitness", None),
                    "fine_rmse": fine.get("best_rmse", None),
                    "fine_nb_inliers_correspondence_set": fine.get("nb_inliers_correspondence_set", None),
                }
            )

        df_median = pd.DataFrame(data_median)
        print("Overall Stats DataFrame (Median):")
        print(df_median)
        save_path = "gedi_data/working_data/stats_results/"
        os.makedirs(save_path, exist_ok=True)  # Create directories if they don’t exist

        df_median.to_csv(os.path.join(save_path, "overall_stats_median.csv"), index=False)


if __name__ == "__main__":
    N = 314  # Maximum number of CAD files to process
    ScanIdentification2.whole_pipeline1(preprocess_scan=False, preprocess_cad=False, compute_eig=True, compute_desc=False)
