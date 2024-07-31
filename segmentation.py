import os
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.neighbors import NearestNeighbors


def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)


def save_point_cloud(pcd, filename):
    o3d.io.write_point_cloud(filename, pcd)


def save_segments(points, labels, output_folder):
    unique_labels = np.unique(labels)
    segment_index = 1

    for label in unique_labels:
        segment_points = points[labels == label]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(segment_points)
        save_point_cloud(
            pcd, os.path.join(output_folder, f"segment_{segment_index}.ply")
        )
        segment_index += 1


def run_dbscan(points, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(points)


def run_mean_shift(points):
    bandwidth = NearestNeighbors(radius=1).fit(points).kneighbors(points, 2)[0].mean()
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    return ms.fit_predict(points)


def run_euclidean_clustering(points, tolerance, min_cluster_size):
    labels = np.zeros(len(points))
    cluster_index = 1

    while np.count_nonzero(labels == 0) > 0:
        seed_index = np.random.choice(np.where(labels == 0)[0])
        seed_point = points[seed_index]
        cluster = [seed_index]
        queue = [seed_index]

        while queue:
            current_index = queue.pop(0)
            neighbors = np.where(
                np.linalg.norm(points - points[current_index], axis=1) < tolerance
            )[0]
            for neighbor in neighbors:
                if labels[neighbor] == 0:
                    labels[neighbor] = cluster_index
                    cluster.append(neighbor)
                    queue.append(neighbor)

        if len(cluster) >= min_cluster_size:
            cluster_index += 1

    return labels


def run_connected_components_labeling(points, connectivity):
    labels = np.zeros(len(points))
    current_label = 1

    for i, point in enumerate(points):
        if labels[i] == 0:
            neighbors = [i]
            while neighbors:
                current_point = neighbors.pop(0)
                labels[current_point] = current_label
                current_neighbors = np.where(
                    np.linalg.norm(points - points[current_point], axis=1)
                    < connectivity
                )[0]
                neighbors.extend(
                    [
                        neighbor
                        for neighbor in current_neighbors
                        if labels[neighbor] == 0
                    ]
                )
            current_label += 1

    return labels


def calculate_min_max_points(points, eps):
    nbrs = NearestNeighbors(n_neighbors=None, radius=eps).fit(points)
    distances, indices = nbrs.radius_neighbors(points)

    num_points_in_radius = [len(ind) for ind in indices]

    return min(num_points_in_radius), max(num_points_in_radius)


def main(input_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    points = load_point_cloud(input_file)

    eps = 20
    min_samples = 100
    min_points, max_points = calculate_min_max_points(points, eps)
    print("Min points in radius:", min_points)
    print("Max points in radius:", max_points)

    dbscan_labels = run_dbscan(points, eps, min_samples)
    print("Number of clusters:", len(np.unique(dbscan_labels)))
    save_segments(points, dbscan_labels, output_folder)

    # Uncomment the following sections to run additional clustering methods

    # Mean Shift
    # mean_shift_labels = run_mean_shift(points)
    # save_segments(points, mean_shift_labels, output_folder)

    # Euclidean Clustering
    # tolerance = 0.1
    # min_cluster_size = 50
    # euclidean_labels = run_euclidean_clustering(points, tolerance, min_cluster_size)
    # save_segments(points, euclidean_labels, output_folder)

    # Connected Components Labeling
    # connectivity = 0.1
    # connected_components_labels = run_connected_components_labeling(points, connectivity)
    # save_segments(points, connected_components_labels, output_folder)


if __name__ == "__main__":
    input_file = "data/fusion/fusion_result.ply"
    output_folder = "data/fusion/segments"
    main(input_file, output_folder)
