import open3d as o3d
import numpy as np
import os
import numpy as np
from gedi import GeDi
import copy
import time
from tqdm import tqdm
import torch
import os
import open3d as o3d
import numpy as np
import csv


class ScanIdentification2:
    def __init__(self, base_path="gedi_data/working_data"):
        # Define the base path
        self.base_path = base_path

        # Construct and create the required subdirectories
        # data
        self.scan_folder = self._create_directory("scan/point_cloud")
        self.cad_folder = self._create_directory("cad/point_cloud")
        # preprocess data
        self.scan_preprocess_folder = self._create_directory("scan/point_cloud/preprocess")
        self.cad_preprocess_folder = self._create_directory("cad/point_cloud/preprocess")

        # descriptors and eigenvalues
        self.scan_desc_folder = self._create_directory("scan/desc")
        self.cad_desc_folder = self._create_directory("cad/desc")
        self.scan_eig_folder = self._create_directory("scan/eigs")
        self.cad_eig_folder = self._create_directory("cad/eigs")

    def _create_directory(self, relative_path: str) -> str:
        """Constructs and creates the directory at base_path/relative_path, then returns the full path."""
        full_path = os.path.join(self.base_path, relative_path)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def __repr__(self):
        return f"ScanIdentification2(base_path={self.base_path})"

    def preprocess_scan(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Scan preprocessing: centering the point cloud."""
        # Centering: subtract the mean of the points.
        points = np.asarray(pcd.points)
        original_centroid = np.mean(points, axis=0)
        centered_points = points - original_centroid
        pcd.points = o3d.utility.Vector3dVector(centered_points)

        return pcd

    def preprocess_cad(self, pcd: o3d.geometry.PointCloud, scale_factor: float) -> o3d.geometry.PointCloud:
        """CAD preprocessing: scaling and centering the point cloud."""
        orig_points = np.asarray(pcd.points)
        scaled_points = orig_points * scale_factor
        pcd.points = o3d.utility.Vector3dVector(scaled_points)
        new_points = np.asarray(pcd.points)
        new_centroid = np.mean(new_points, axis=0)
        centered_points = new_points - new_centroid
        pcd.points = o3d.utility.Vector3dVector(centered_points)

        return pcd

    def preprocess_all_scans(self):
        """
        Processes every scan in self.scan_folder and saves the preprocessed point clouds
        in a new subfolder named 'preprocess' inside the scan folder.
        """
        output_folder = os.path.join(self.scan_folder, "preprocess")
        os.makedirs(output_folder, exist_ok=True)

        # Filter for .ply files
        scan_files = [f for f in os.listdir(self.scan_folder) if f.lower().endswith(".ply")]

        # Use tqdm for progress indication
        for filename in tqdm(scan_files, desc="Preprocessing scans", unit="scan"):
            full_path = os.path.join(self.scan_folder, filename)
            pcd = o3d.io.read_point_cloud(full_path)
            preprocessed_pcd = self.preprocess_scan(pcd)
            output_path = os.path.join(output_folder, filename)
            o3d.io.write_point_cloud(output_path, preprocessed_pcd)

    def preprocess_all_cads(self, scale_factor: float):
        """
        Processes every CAD in self.cad_folder and saves the preprocessed point clouds
        in a new subfolder named 'preprocess' inside the CAD folder.
        """
        output_folder = os.path.join(self.cad_folder, "preprocess")
        os.makedirs(output_folder, exist_ok=True)

        cad_files = [f for f in os.listdir(self.cad_folder) if f.lower().endswith(".ply")]

        for filename in tqdm(cad_files, desc="Preprocessing CADs", unit="cad"):
            full_path = os.path.join(self.cad_folder, filename)
            pcd = o3d.io.read_point_cloud(full_path)
            preprocessed_pcd = self.preprocess_cad(pcd, scale_factor)
            output_path = os.path.join(output_folder, filename)
            o3d.io.write_point_cloud(output_path, preprocessed_pcd)

    def load_model(self):
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

    def compute_and_save_desc(self, model):
        """
        Compute and save descriptors for a limited number of scans and CADs.
        """

        def _process_folder(input_folder, output_folder, descriptor_type):
            total_time = 0
            # Process only the first 'count' files
            for file in tqdm(os.listdir(input_folder), desc=f"Computing {descriptor_type} descriptors"):
                start = time.time()
                file_path = os.path.join(input_folder, file)

                # Compute descriptors; assuming compute_descriptors is a @staticmethod of ScanIdentification2
                desc = self.compute_descriptors(file_path, model)

                # Create output file path with a .npy extension
                base_name = os.path.splitext(file)[0]
                out_path = os.path.join(output_folder, f"{base_name}.npy")

                np.save(out_path, desc)
                total_time += time.time() - start
            return total_time

        # Process scans and CADs
        total_time_scan = _process_folder(self.scan_preprocess_folder, self.scan_desc_folder, "scan")
        total_time_cad = _process_folder(self.cad_preprocess_folder, self.cad_desc_folder, "cad")

        # Print timing statistics
        print("Time taken for computing descriptors + IO operations:")
        print("------------------------------------------------------")
        print(f"Total time for scan descriptors: {total_time_scan:.2f} seconds")
        print(f"Total time for cad descriptors: {total_time_cad:.2f} seconds")
        print(f"Total time: {total_time_scan + total_time_cad:.2f} seconds")
        print(f"Average time per scan: {total_time_scan / 5:.2f} seconds")
        print(f"Average time per cad: {total_time_cad / 5:.2f} seconds")

    def compute_and_save_eig(self):
        start = time.time()
        print("Computing and saving eigenvalues")

        for scan in tqdm(os.listdir(self.scan_preprocess_folder), desc="Computing scan eigenvalues"):
            scan_path = os.path.join(self.scan_preprocess_folder, scan)
            pcd = o3d.io.read_point_cloud(scan_path)
            eig = np.linalg.eig(np.cov(np.asarray(pcd.points).T))[0]
            np.save(os.path.join(self.scan_eig_folder, scan.split(".")[0]), eig)

        for cad in tqdm(os.listdir(self.cad_preprocess_folder), desc="Computing cad eigenvalues"):
            cad_path = os.path.join(self.cad_preprocess_folder, cad)
            pcd = o3d.io.read_point_cloud(cad_path)
            eig = np.linalg.eig(np.cov(np.asarray(pcd.points).T))[0]
            np.save(os.path.join(self.cad_eig_folder, cad.split(".")[0]), eig)

        end = time.time()
        print("Done !")
        print(f"Time taken for computing and saving eigenvalues: {end - start:.2f} seconds")

    def coarse_filtering(self, scan_path):
        cad_score = {}
        cad_eig_dict = {}
        start = time.time()
        target_scan = os.path.basename(scan_path).split(".")[0]

        # Load scan eigenvalues and sort them.
        scan_eig = np.load(os.path.join(self.scan_eig_folder, target_scan + ".npy"))
        scan_eig = np.sort(scan_eig)[::-1]

        print("\033[93m🛠️ ----------------- Coarse Filtering / Eigenvalues Filtering (In Progress...) ----------------- 🛠️\033[0m")
        print(f"Target scan : {target_scan}")
        print(f"Target scan eigenvalues : {', '.join(f'{val:.4f}' for val in scan_eig[:3])}")

        # Compute similarity between scan and CADs.
        for cad in tqdm(os.listdir(self.cad_eig_folder), desc="Processing cads for coarse filtering"):
            cad_num = cad.split(".")[0]
            # Load CAD eigenvalues and sort them.
            cad_eig_val = np.load(os.path.join(self.cad_eig_folder, cad))
            cad_eig_val = np.sort(cad_eig_val)[::-1]

            # Compute similarity.
            similarity = ScanIdentification2.eig_similarity(scan_eig, cad_eig_val, penalty=True)
            cad_score[cad_num] = similarity
            cad_eig_dict[cad_num] = cad_eig_val

        sorted_cad_score = dict(sorted(cad_score.items(), key=lambda item: item[1]))
        end = time.time()
        print(f"⏳ Took : {end - start:.2f} seconds")

        # Display results
        print("\nSorted CADs by Eigenvalue Similarity:")
        header = f"| {'Rank':^6} | {'CAD':^10} | {'Score (lower is better)':^25} | {'Eigenvalues':^30} | {'Status':^8} |"
        print(header)
        print("-" * len(header))
        for idx, (cad, score) in enumerate(sorted_cad_score.items(), start=1):
            eig_values = ", ".join(f"{val:.4f}" for val in cad_eig_dict[cad][:3])
            status = ""
            if cad == target_scan:
                status = "✅" if idx == 1 else "⬅️ "
            print(f"| {idx:^6} | {cad:^10} | {score:^25.4f} | {eig_values:^30} | {status:^8} |")
        print("-" * len(header))
        print("\n\n")
        print("\033[92m✅ ----------------- Coarse Filtering / Eigenvalues Filtering (Done!) ----------------- ✅\033[0m")

        # --- Write the results to a CSV file ---
        results_folder = os.path.join(self.base_path, "results/csv")
        os.makedirs(results_folder, exist_ok=True)
        # Create target-specific folder
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
        return sorted_cad_score

    def fine_filtering(self, scan_path, candidate_cads):
        print("\033[93m🛠️ ----------------- Fine Filtering / Registration (In progress...) ----------------- 🛠️\033[0m")
        start = time.time()

        # Load the scan descriptors and scan point cloud.
        scan_basename = os.path.basename(scan_path).split(".")[0]
        target_scan = scan_basename  # target scan number
        scan_desc = np.load(os.path.join(self.scan_desc_folder, scan_basename + ".npy"))
        scan_pts = o3d.io.read_point_cloud(scan_path).points
        pcd0_dsdv = o3d.pipelines.registration.Feature()
        pcd0_dsdv.data = scan_desc.T

        registration_result = {}

        # --- Create an output folder for the combined point clouds ---
        # Use the target-specific folder for results.
        results_folder = os.path.join(self.base_path, "results/csv")
        os.makedirs(results_folder, exist_ok=True)
        target_folder = os.path.join(results_folder, f"{target_scan}_exp")
        os.makedirs(target_folder, exist_ok=True)
        # Also, create a subfolder for visualization (combined PCDs)
        viz_folder = os.path.join(target_folder, "viz_results")
        os.makedirs(viz_folder, exist_ok=True)

        for cad in tqdm(candidate_cads, desc="Processing cads for fine filtering"):
            # Load the CAD descriptors and CAD point cloud.
            cad_desc = np.load(os.path.join(self.cad_desc_folder, cad + ".npy"))
            cad_pts = o3d.io.read_point_cloud(os.path.join(self.cad_preprocess_folder, cad + ".ply")).points

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
                max_correspondence_distance=0.02,  # can be turn to 0.02 is 2 cm
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,  # can be turn to 3 or 4, higher value will take more time
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.02),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 3000),
            )

            # Get number of inliers if available.
            num_inliers = len(est_result.correspondence_set) if hasattr(est_result, "correspondence_set") else "N/A"

            registration_result[cad] = {
                "fitness": est_result.fitness,
                "inlier_rmse": est_result.inlier_rmse,
                "transformation": est_result.transformation,
                "num_inliers": num_inliers,
            }

            # --- Create combined colored point clouds for visualization ---
            pcd_scan = _pcd0  # Original scan point cloud
            pcd_cad = _pcd1  # Original candidate CAD point cloud

            pcd_scan_copy = copy.deepcopy(pcd_scan)
            pcd_scan_copy.transform(est_result.transformation)
            pcd_scan_copy.paint_uniform_color([1, 0.706, 0])  # Yellow
            pcd_cad_copy = copy.deepcopy(pcd_cad)
            pcd_cad_copy.paint_uniform_color([0, 0.651, 0.929])  # Light Blue
            combined_pcd = pcd_scan_copy + pcd_cad_copy
            output_filename = os.path.join(viz_folder, f"{cad}_aligned.ply")
            o3d.io.write_point_cloud(output_filename, combined_pcd)

        end = time.time()
        print(f"⏳ Took : {end - start:.2f} seconds")

        # --- Process and Print Sorted & Filtered Registration Results ---
        inlier_threshold = 0.02
        filtered_results = [
            (cad, data["fitness"], data["inlier_rmse"], data["num_inliers"])
            for cad, data in registration_result.items()
            if data["inlier_rmse"] <= inlier_threshold
        ]

        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

        print("\nSorted Registration Results (filtered by inlier_rmse <= {:.2f}):".format(inlier_threshold))
        header = "| {0:^5} | {1:^15} | {2:^10} | {3:^15} | {4:^12} | {5:^12} |".format(
            "Rank", "CAD", "Fitness", "Inlier RMSE", "Num Inliers", "Status"
        )
        print(header)
        print("-" * len(header))
        for rank, (cad, fitness, inlier_rmse, num_inliers) in enumerate(sorted_results, start=1):
            if cad == target_scan:
                status = "✅ " if rank == 1 else "⬅️ "
            else:
                status = ""
            print(
                "| {0:^5} | {1:^15} | {2:^10.4f} | {3:^15.4f} | {4:^12} | {5:^12} |".format(
                    rank, cad, fitness, inlier_rmse, num_inliers, status
                )
            )
        print("-" * len(header))
        print("\n")
        print("\033[92m✅ ----------------- Fine Filtering (Done!) ----------------- ✅\033[0m")

        # --- Write Fine Filtering results to CSV ---
        csv_file_path_fine = os.path.join(target_folder, "fine_filtering_results.csv")
        with open(csv_file_path_fine, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Rank", "CAD", "Fitness", "Inlier RMSE", "Num Inliers", "Status"])
            for rank, (cad, fitness, inlier_rmse, num_inliers) in enumerate(sorted_results, start=1):
                if cad == target_scan:
                    status = "✅" if rank == 1 else "⬅️"
                else:
                    status = ""
                writer.writerow([rank, cad, f"{fitness:.4f}", f"{inlier_rmse:.4f}", num_inliers, status])
        print(f"CSV file saved to: {csv_file_path_fine}")

        return registration_result

    @staticmethod
    def eig_similarity(scan_eig, cad_eig, penalty=True):
        # Compute similarity between scan and cad eigenvalues by % difference and sum the %
        # Add a penalty for compensating eigenvalues
        diff = np.abs(scan_eig - cad_eig) / scan_eig
        penalty = np.std(scan_eig) if penalty else 0
        return np.sum(diff) + penalty

    @staticmethod
    def compute_descriptors(pcd_path: str, model: GeDi, voxel_size: float = 0.001, patches_per_pair: int = 5000):
        """
        Compute a descriptor for the given point cloud file using the GeDi model.
        """
        # Load the point cloud from file
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)

        if points.shape[0] == 0:
            raise ValueError("The point cloud contains no points.")

        # Randomly sample points (patches) from the original point cloud
        inds = np.random.choice(points.shape[0], patches_per_pair, replace=True)
        pts_sample = points[inds]
        pts_tensor = torch.tensor(pts_sample).float()

        # Downsample the point cloud and estimate normals
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.estimate_normals()
        down_points = np.asarray(pcd_down.points)
        pcd_down_tensor = torch.tensor(down_points).float()

        # Compute the descriptor using the provided GeDi model
        descriptor = model.compute(pts=pts_tensor, pcd=pcd_down_tensor)
        return descriptor

    @staticmethod
    def get_next_experiment_folder(parent_folder):
        """
        Checks the parent_folder for existing subfolders with names matching "exp<number>"
        and returns the path for the next experiment folder (e.g., if exp1 and exp2 exist, returns exp3).
        """
        import re

        # List directories in parent_folder that match "exp" followed by digits
        existing = [
            d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d)) and re.match(r"^exp\d+$", d)
        ]
        if existing:
            numbers = [int(re.findall(r"\d+", d)[0]) for d in existing]
            next_number = max(numbers) + 1
        else:
            next_number = 1
        new_folder = os.path.join(parent_folder, f"exp{next_number}")
        os.makedirs(new_folder, exist_ok=True)
        return new_folder


if __name__ == "__main__":
    # Initialize the ScanIdentification2 instance with appropriate base folder
    identification = ScanIdentification2(base_path="gedi_data/working_data")

    # identification.preprocess_all_scans()  # centering
    # identification.preprocess_all_cads(scale_factor=0.001)  # scaling and centering

    cum_error_threshold = 2.3
    gedi = identification.load_model()

    # pre-compute eig and desc (only needed if not already computed)
    # identification.compute_and_save_desc(model=gedi)
    # identification.compute_and_save_eig()

    scans = ["763638.ply"]

    for scan in scans:
        # Process coarse fitering
        scan_path = os.path.join(identification.scan_preprocess_folder, scan)
        cad_score = identification.coarse_filtering(scan_path)

        # Process fine filtering
        candidate_cads = [cad for cad, score in cad_score.items() if score < cum_error_threshold]
        identification.fine_filtering(scan_path, candidate_cads)
