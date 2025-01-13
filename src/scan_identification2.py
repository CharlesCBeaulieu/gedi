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
import json
from tqdm import tqdm


class ScanIdentification2:
    def __init__(
        self, scan_folder, scan_desc_folder, scan_eig_folder, cad_folder, cad_desc_folder, cad_eig_folder, metadata_path
    ):
        self.scan_folder = scan_folder
        self.scan_desc_folder = scan_desc_folder
        self.scan_eig_folder = scan_eig_folder
        self.cad_folder = cad_folder
        self.cad_desc_folder = cad_desc_folder
        self.cad_eig_folder = cad_eig_folder
        self.metadata_path = metadata_path

    @staticmethod
    def compute_descriptors(pcd_path: str, model: GeDi, preprocess):
        voxel_size = 0.05
        patches_per_pair = 5000

        # Load ply and process if needed
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd = preprocess(pcd) if preprocess else pcd

        # Downsample and estimate normals
        inds = np.random.choice(np.asarray(pcd.points).shape[0], patches_per_pair, replace=True)
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
    def compute_and_save_desc(gedi):
        total_time_scan = 0
        total_time_cad = 0

        # Compute and save descriptors for scans
        for scan in tqdm(os.listdir(identification.scan_folder), desc="Computing scan descriptors"):
            start = time.time()
            scan_path = os.path.join(identification.scan_folder, scan)
            scan_desc = ScanIdentification2.compute_descriptors(scan_path, gedi, ScanIdentification2.preprocess_scan)
            scan_desc_path = os.path.join(identification.scan_desc_folder, scan.split(".")[0])
            np.save(scan_desc_path, scan_desc)
            end = time.time()
            t = end - start
            total_time_scan += t

        # Compute and save descriptors for cads
        for cad in tqdm(os.listdir(identification.cad_folder), desc="Computing cad descriptors"):
            start = time.time()
            cad_path = os.path.join(identification.cad_folder, cad)
            cad_desc = ScanIdentification2.compute_descriptors(cad_path, gedi, ScanIdentification2.preprocess_cad)
            cad_desc_path = os.path.join(identification.cad_desc_folder, cad.split(".")[0])
            np.save(cad_desc_path, cad_desc)
            end = time.time()
            t = end - start
            total_time_cad += t

        print("Time taken for computing of the dest + IO opperations for computing the descriptors")
        print("------------------------------------------------------")
        print(f"Total time for scan descriptors: {total_time_scan}")
        print(f"Total time for cad descriptors: {total_time_cad}")
        print(f"Total time: {total_time_scan + total_time_cad}")
        print(f"Average time per scan: {total_time_scan / len(os.listdir(identification.scan_folder))}")
        print(f"Average time per cad: {total_time_cad / len(os.listdir(identification.cad_folder))}")

    @staticmethod
    def compute_and_save_eig():
        for scan in os.listdir(identification.scan_folder):
            scan_path = os.path.join(identification.scan_folder, scan)
            pcd = o3d.io.read_point_cloud(scan_path)
            pcd = ScanIdentification2.preprocess_scan(pcd)
            eig = np.linalg.eig(np.cov(np.asarray(pcd.points).T))[0]
            np.save(f"gedi_data/eig/{scan.split('.')[0]}", eig)


if __name__ == "__main__":
    # Initialize the ScanIdentification2 instance with appropriate folders and metadata path
    identification = ScanIdentification2(
        scan_folder="gedi_data/gen_scan",
        scan_desc_folder="gedi_data/gen_scan_desc",
        scan_eig_folder="gedi_data/gen_scan_eig",
        cad_folder="gedi_data/gt_cad",
        cad_desc_folder="gedi_data/gt_cad_desc",
        cad_eig_folder="gedi_data/gt_cad_eig",
        metadata_path="gedi_data/metadata.json",
    )
    gedi = ScanIdentification2.load_model()

    # only need to run this line if the descriptors are not already computed
    # identification.compute_and_save_desc(gedi)
