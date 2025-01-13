import open3d as o3d
import numpy as np
import os
import numpy as np
import open3d as o3d
from gedi import GeDi
import torch
import matplotlib.pyplot as plt
import copy
import time
from mpl_toolkits.mplot3d import Axes3D


# The goal of this script is to identify the scans that are present in the data pool
# first step is to use a coarse method to filter most of the scans and then use
# a more fine method to identify the scans that are present in the data pool


class ScanIdentification:
    def __init__(self, scan_path, cad_path, output_path):
        self.scan_path = scan_path
        self.cad_path = cad_path
        self.output_path = output_path

    @staticmethod
    def load_scan_and_process(scan_path: str) -> o3d.geometry.PointCloud:
        pcd = o3d.io.read_point_cloud(scan_path)

        # Center the scan
        centroid = np.mean(pcd.points, axis=0)
        centered_scan_points = pcd.points - centroid

        pcd.points = o3d.utility.Vector3dVector(centered_scan_points)
        return pcd

    @staticmethod
    def coarse_identification(scan_pcd: o3d.geometry.PointCloud, scan_id: str, cads_folder: str, voxel_size: float) -> dict:
        """
        This function will compare the eigenvalues of the scan to the eigenvalues of the cads
        and return the difference between the eigenvalues. The smallest difference will be the
        most likely candidate for the scan.
        """
        # step 1: load the scan and compute the eigenvalues
        # TODO this step could be better done with more features than just the eigenvalues
        # TODO for example, could use pointnet to extract features from the cads...
        # need to voxel downsample to ignore density of points and focus on shape
        scan_down = scan_pcd.voxel_down_sample(voxel_size)

        scan_eigs = np.linalg.eigvals(np.cov(np.asarray(scan_down.points).T))
        sorted_scan_eigs = np.sort(scan_eigs)[::-1]  # sort the eigenvalues, biggest first

        results = {}
        for cad_file in os.listdir(cads_folder):
            cad_num = cad_file.split(".")[0]
            # step 2: load the cad and compute the eigenvalues
            cad_pcd = o3d.io.read_point_cloud(os.path.join(cads_folder, cad_file))
            cad_pcd.points = o3d.utility.Vector3dVector(np.asarray(cad_pcd.points) / 1000)
            cad_pcd_down = cad_pcd.voxel_down_sample(voxel_size)
            cad_eigs = np.linalg.eigvals(np.cov(np.asarray(cad_pcd_down.points).T))
            sorted_cad_eigs = np.sort(cad_eigs)[::-1]  # sort the eigenvalues, biggest first

            # compare the eigenvalues (I use the norm of the difference but could use other metrics)
            # TODO: could use other metrics to compare the eigenvalues
            diff = np.linalg.norm(sorted_scan_eigs - sorted_cad_eigs)
            results[cad_num] = diff

        # step 3 : sort the results and return the dict
        results = dict(sorted(results.items(), key=lambda item: item[1]))

        # interpretation
        # check if the scan is in the results, and what is the rank
        print("############################# Coarse results #######################################")
        if scan_id in results:
            rank = list(results.keys()).index(scan_id)
            print(f"Scan {scan_id} is in the results with rank {rank}")
        else:
            print(f"Scan {scan_id} is not in the results for coarse filtering")
        return results, rank

    @staticmethod
    def gedi_compute(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
    ) -> dict:
        config = {
            "dim": 32,  # descriptor output dimension
            "samples_per_batch": 500,  # batches to process the data on GPU
            "samples_per_patch_lrf": 4000,  # num. of point to process with LRF
            "samples_per_patch_out": 512,  # num. of points to sample for pointnet++
            "r_lrf": 0.5,  # LRF radius
            "fchkpt_gedi_net": "data/chkpts/3dmatch/chkpt.tar",
        }  # path to checkpoint

        voxel_size = 0.05
        patches_per_pair = 5000

        # initialising class
        gedi = GeDi(config=config)

        # getting a pair of point clouds
        pcd0 = source
        pcd1 = target

        pcd0.points = o3d.utility.Vector3dVector(np.asarray(pcd0.points) / 1000)
        pcd1.points = o3d.utility.Vector3dVector(np.asarray(pcd1.points) / 1000)
        # pcd0 = o3d.io.read_point_cloud('data/assets/threed_match_7-scenes-redkitchen_cloud_bin_0.ply')
        # pcd1 = o3d.io.read_point_cloud('data/assets/threed_match_7-scenes-redkitchen_cloud_bin_5.ply')

        # estimating normals (only for visualisation)
        pcd0.estimate_normals()
        pcd1.estimate_normals()

        # randomly sampling some points from the point cloud
        inds0 = np.random.choice(np.asarray(pcd0.points).shape[0], patches_per_pair, replace=True)
        inds1 = np.random.choice(np.asarray(pcd1.points).shape[0], patches_per_pair, replace=True)

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
            max_correspondence_distance=0.02,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.02),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000),
        )

        return est_result

    @staticmethod
    def fine_identification(scan: o3d.geometry.PointCloud, scan_id: str, candidates: dict) -> dict:
        """
        args:
        - scan : o3d.geometry.PointCloud : the scan to identify
        - candidate : dict : the candidate to compare to the scan (the output of coarse_identification)
        """
        total_time = 0
        # try to align the scan with the candidate
        # the one with the best alignment is the most likely to be the cad we are looking for
        for idx, candidate_id in enumerate(candidates):
            start = time.time()

            # make a copy of the scan to avoid modifying the original
            scan_c = copy.deepcopy(scan)

            # load the candidate and center it
            candidate_pcd = o3d.io.read_point_cloud(os.path.join(CADS_FOLDER, candidate_id + ".ply"))
            candidate_points = np.asarray(candidate_pcd.points)
            candidate_mean = np.mean(candidate_points, axis=0)
            candidate_pcd.points = o3d.utility.Vector3dVector(candidate_points - candidate_mean)
            candidate_pcd.points = o3d.utility.Vector3dVector(np.asarray(candidate_pcd.points) / 1000)

            # compute the transformation using GeDi
            gedi_result = ScanIdentification.gedi_compute(scan_c, candidate_pcd)

            stop = time.time()
            iteration_time = stop - start
            total_time += iteration_time

            # save the result as a png
            print(
                f"Time for iteration {idx} : {iteration_time:.2f}"
            )  # time only include the network computation, not the plotting
            save_result_png(
                scan_c,
                scan_id,
                candidate_pcd,
                candidate_id,
                gedi_result,
                f"{OUTPUT_FOLDER}/scan{scan_id}_cadidate{candidate_id}.png",
            )

            if scan_id == candidate_id:
                # transform the candidate to the scan
                candidate_pcd_trans = candidate_pcd.transform(gedi_result.transformation)

                # color the points
                candidate_pcd_trans.paint_uniform_color([0, 0, 1])
                scan_c.paint_uniform_color([1, 0, 0])

                # Combine points and colors
                combined_points = np.vstack((np.asarray(scan_c.points), np.asarray(candidate_pcd_trans.points)))
                combined_colors = np.vstack((np.asarray(scan_c.colors), np.asarray(candidate_pcd_trans.colors)))

                # Create new point cloud with combined points and colors and save it
                res_pcd = o3d.geometry.PointCloud()
                res_pcd.points = o3d.utility.Vector3dVector(combined_points)
                res_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
                o3d.io.write_point_cloud(f"{OUTPUT_FOLDER}/match_{scan_id}.ply", res_pcd)

        print(f"Total time for fine identification : {total_time:.2f}")

    @staticmethod
    def compute_descriptors():
        scan_files = [f for f in os.listdir(self.scan_path) if f.endswith('.ply')]
        cad_files = [f for f in os.listdir(self.cad_path) if f.endswith('.ply')]

        descriptors = {}

        for scan_file in scan_files:
            scan_id = scan_file.split('.')[0]
            scan_pcd = o3d.io.read_point_cloud(os.path.join(self.scan_path, scan_file))
            scan_pcd.points = o3d.utility.Vector3dVector(np.asarray(scan_pcd.points) / 1000)

            for cad_file in cad_files:
            cad_id = cad_file.split('.')[0]
            cad_pcd = o3d.io.read_point_cloud(os.path.join(self.cad_path, cad_file))
            cad_pcd.points = o3d.utility.Vector3dVector(np.asarray(cad_pcd.points) / 1000)

            gedi_result = self.gedi_compute(scan_pcd, cad_pcd)

            descriptors[scan_id] = {
                "scan_path": os.path.join(self.scan_path, scan_file),
                "cad_path": os.path.join(self.cad_path, cad_file),
                "descriptor": gedi_result.transformation.tolist()
            }

        with open(os.path.join(self.output_path, 'descriptors.json'), 'w') as f:
            json.dump(descriptors, f, indent=4)

    
def save_result_png(scan, scan_id, cad, cad_id, result, output_file):
    # Extract points for plotting
    points_scan = np.asarray(scan.points)
    points_cad = np.asarray(cad.points)

    # Apply transformation to cad
    transformed_cad = copy.deepcopy(cad).transform(result.transformation)
    points_transformed_cad = np.asarray(transformed_cad.points)

    # Create the figure
    fig = plt.figure(figsize=(12, 10))

    # Subplot 1: scan before transformation
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.scatter(
        points_scan[:, 0],
        points_scan[:, 1],
        points_scan[:, 2],
        c="red",
        s=1,
        label="Scan",
    )
    ax1.set_title(f"Scan {scan_id}")
    ax1.legend()
    ax1.set_box_aspect(
        [
            np.ptp(points_scan[:, 0]),
            np.ptp(points_scan[:, 1]),
            np.ptp(points_scan[:, 2]),
        ]
    )

    # Subplot 2: cad before transformation
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.scatter(points_cad[:, 0], points_cad[:, 1], points_cad[:, 2], c="blue", s=1, label="Cad")
    ax2.set_title(f"Cad {cad_id}")
    ax2.legend()
    ax2.set_box_aspect([np.ptp(points_cad[:, 0]), np.ptp(points_cad[:, 1]), np.ptp(points_cad[:, 2])])

    # Subplot 3: scan and transformed cad
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.scatter(
        points_scan[:, 0],
        points_scan[:, 1],
        points_scan[:, 2],
        c="red",
        s=1,
        label="Scan",
    )
    ax3.scatter(
        points_transformed_cad[:, 0],
        points_transformed_cad[:, 1],
        points_transformed_cad[:, 2],
        c="blue",
        s=1,
        label="Transformed Cad",
    )
    ax3.set_title(f"Scan {scan_id} and Transformed Cad {cad_id}")
    ax3.legend()
    ax3.set_box_aspect(
        [
            np.ptp(points_scan[:, 0]),
            np.ptp(points_scan[:, 1]),
            np.ptp(points_scan[:, 2]),
        ]
    )

    # Subplot 4: Registration result as text
    ax4 = fig.add_subplot(2, 2, 4)
    fitness = result.fitness
    inlier_rmse = result.inlier_rmse
    correspondence_set_size = len(result.correspondence_set)
    result_text = (
        f"Registration Result:\n"
        f"Fitness: {fitness:.2e}\n"
        f"Inlier RMSE: {inlier_rmse:.2e}\n"
        f"Correspondence Set Size: {correspondence_set_size}"
    )
    ax4.text(0.5, 0.5, result_text, ha="center", va="center", fontsize=12, wrap=True)
    ax4.set_title("Result Summary")
    ax4.axis("off")  # Remove axes for better text display

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Figure saved as '{output_file}'.")
    plt.close(fig)


def plot_coarse_results(ranks: list, output_folder: str):
    # plot the results
    # cumulative distribution of the results
    plt.hist(ranks, bins=20, color="skyblue", edgecolor="black", alpha=0.7, density=True)

    # Fit a normal distribution to the data
    mu, std = np.mean(ranks), np.std(ranks)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-((x - mu) ** 2) / (2 * std**2)) / (std * np.sqrt(2 * np.pi))

    # Plot the normal distribution
    plt.plot(x, p, "k", linewidth=2)

    # Calculate the 90th percentile
    perc_90 = np.percentile(ranks, 90)
    plt.axvline(perc_90, color="r", linestyle="dashed", linewidth=1)
    plt.text(perc_90, max(p) * 0.9, "90th percentile", color="r")

    plt.title(f"Ranks for the coarse identification")
    plt.xlabel("Rank")
    plt.ylabel("Density")
    plt.savefig(f"{output_folder}/coarse_results.png")
    plt.close()


if __name__ == "__main__":
    scan_identifier = ScanIdentification(
        scan_path="data/generated_SCAN_pool/pcd",
        cad_path="data/sbi_CAD",
        output_path="results/scan_identification",
    )
    SCANS_FOLDER = "data/generated_SCAN_pool/pcd"
    CADS_FOLDER = "data/sbi_CAD"
    OUTPUT_FOLDER = "results/scan_identification"
    SCAN_FILES_LIST = [files for files in os.listdir(SCANS_FOLDER) if files.endswith(".ply")]
    TopCandidateLimit = 20
    ranks = []

    # check in the output folder exists
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    # experiment number
    files = os.listdir(OUTPUT_FOLDER)
    exp_num = len(files) + 1
    OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, f"exp_{exp_num}")
    os.makedirs(OUTPUT_FOLDER)

    for scan_file in SCAN_FILES_LIST[:30]:
        scan_id = scan_file.split(".")[0]
        # load the scan and scale it to fit the cads domain
        scan = ScanIdentification.load_scan_and_process(scan_path=os.path.join(SCANS_FOLDER, scan_file))
        # coarse identification
        coarse_results, rank = ScanIdentification.coarse_identification(
            scan_pcd=scan, scan_id=scan_id, cads_folder=CADS_FOLDER, voxel_size=0.01
        )
        ranks.append(rank)
        # fine identification using the top candidates
        candidates = list(coarse_results.keys())[:TopCandidateLimit]
        fine_results = ScanIdentification.fine_identification(scan=scan, scan_id=scan_id, candidates=candidates)

    plot_coarse_results(ranks, OUTPUT_FOLDER)
