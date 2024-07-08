"""
The KMeans class within this module can load data from a specified file,
run the K-Means clustering algorithm, and save the cluster assignments
to an output file. Additionally, it can generate a plot visualizing the
clustered data and centroids.
"""
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    """
    Attributes:
        cluster_number (int): The number of clusters to form.
        input_file_path (str): Path to the input data file.
        output_file_path (Optional[str]): Path to the output file where cluster
        assignments will be saved.
        image_file_path (Optional[str]): Path to the output image file for the cluster
        visualization.
        centroids (Optional[np.ndarray]): The centroids of the clusters.
        assignments (Optional[np.ndarray]): An array indicating the cluster assignment
        for each data point.
        data (Optional[np.ndarray]): The data points loaded from the input file.
    """

    def __init__(self,
                 cluster_number: int,
                 input_file_path: str,
                 output_file_path: Optional[str] = None,
                 image_file_path: Optional[str] = None) -> None:
        """
        Initializes the KMeans clustering with specified parameters.
        Args:
            cluster_number (int): The number of clusters to form.
            input_file_path (str): Path to the input data file.
            output_file_path (Optional[str]): Path to the output file where cluster
            assignments will be saved.
            image_file_path (Optional[str]): Path to the output image file for the
            cluster visualization.
        """
        self.cluster_number: int = cluster_number
        self.input_file_path: str = input_file_path
        self.output_file_path: Optional[str] = output_file_path
        self.image_file_path: Optional[str] = image_file_path
        self.centroids: np.ndarray = np.empty(0)
        self.assignments: np.ndarray = np.empty(0)
        self.data: np.ndarray = np.empty(0)

    def load_data_from_file(self) -> None:
        """
        Loads data points from a specified file path.

        The method expects the file to contain one data point per line,
        with each component of the data point separated by whitespace.
        """
        data_list: List[List[float]] = []
        with open(self.input_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_components = line.split()
                data_row = [float(component) for component in line_components]
                data_list.append(data_row)
        self.data = np.array(data_list)

    def initialize_centroids_randomly(self) -> None:
        """
        Initializes centroids by randomly selecting `n_clusters`
        data points as the initial centroids.
        """
        if self.data is not None:
            centroid_indexes = np.random.choice(len(self.data), self.cluster_number, replace=False)
            self.centroids = self.data[centroid_indexes]
        else:
            raise ValueError("Data not loaded.")

    def calculate_distances(self) -> np.ndarray:
        """
        Calculates the Euclidean distance from each data point to each centroid.
        """
        if self.centroids is not None and self.data is not None:
            distances = []
            for centroid in self.centroids:
                squared_diffs = (self.data - centroid) ** 2
                squared_dists = squared_diffs.sum(axis=1)
                distances.append(np.sqrt(squared_dists))
            return np.stack(distances, axis=1)

        raise ValueError("Centroids or data not initialized.")

    def assign_data_to_clusters(self) -> None:
        """
        Assigns each data point to the closest centroid to form clusters.
        """
        distances = self.calculate_distances()
        self.assignments = np.argmin(distances, axis=1)

    def update_centroids_based_on_mean(self) -> None:
        """
        Updates the centroids to the mean of the data points assigned to each cluster.
        """
        new_centroids: List[np.ndarray] = []
        for i in range(self.cluster_number):
            if np.any(self.assignments == i):
                cluster_mean = self.data[self.assignments == i].mean(axis=0)
                new_centroids.append(cluster_mean)
            else:
                new_centroids.append(self.centroids[i])
        self.centroids = np.array(new_centroids)

    def centroids_equal(self, old_centroids: np.ndarray) -> bool:
        """
        Checks if the centroids have changed from the previous iteration.
        """
        return np.array_equal(old_centroids, self.centroids)

    def run_k_means(self) -> None:
        """
        Executes the K-Means clustering algorithm.

        The method iteratively assigns data points to clusters
        based on the closest centroid,
        then updates centroids until they no longer change.
        """
        self.load_data_from_file()
        self.initialize_centroids_randomly()

        while True:
            old_centroids = self.centroids.copy()
            self.assign_data_to_clusters()
            self.update_centroids_based_on_mean()

            if self.centroids_equal(old_centroids):
                break

    def save_cluster_assignments(self) -> None:
        """
        Saves the cluster assignments to the specified output file.
        """
        if not self.output_file_path:
            raise ValueError("Output file path not specified.")

        with open(self.output_file_path, 'w', encoding='utf-8') as file:
            file.write(f'{len(self.centroids)}\n')
            for idx, centroid in enumerate(self.centroids):
                file.write(f'{idx} {centroid[0]} {centroid[1]}\n')

    def plot_clusters(self) -> None:
        """
        Plots the clustered data and centroids.
        """
        if not self.image_file_path:
            raise ValueError("Image file path not specified.")
        if self.data is None:
            raise ValueError("No data available for plotting.")

        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.assignments, s=50, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=500, c='red', marker='o')
        plt.title('K-Means Clustering')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.savefig(self.image_file_path)
        plt.show()


def run_script(cluster_number: int,
               input_file_path: str,
               output_file_path: str,
               image_file_path: str) -> None:
    """
    Executes the KMeans clustering algorithm with specified parameters
    and visualizes the results.

    Args:
        cluster_number (int): The number of clusters to form.
        input_file_path (str): Path to the input data file.
        output_file_path (str): Path to the output file where cluster
        assignments will be saved.
        image_file_path (str): Path to the output image file for
        cluster visualization.
    """
    kmeans = KMeans(cluster_number=cluster_number,
                    input_file_path=input_file_path,
                    output_file_path=output_file_path,
                    image_file_path=image_file_path)
    kmeans.run_k_means()
    kmeans.save_cluster_assignments()
    kmeans.plot_clusters()


run_script(4, 'data.txt', 'final_centroids.txt', 'clusters.png')
