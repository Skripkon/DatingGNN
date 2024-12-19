import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import pickle
import networkx as nx


class ClusterCentroidCalculator:
    def __init__(self, clusters_path, graph, clustering_algorithm):
        self.clusters_path = clusters_path
        self.G = graph
        self.clusters = pd.read_csv(clusters_path)
        self.clustering_algorithm = clustering_algorithm

    def compute_cluster_centroid(self, cluster: int):
        """Computes the centroid of a given cluster by calculating the mean position of its vertices."""
        # Get the list of vertices in the given cluster
        vertices = self.clusters[self.clusters["cluster_id"] == cluster]["vertex_id"].astype(str).values
        distance_matrix = np.zeros(shape=(len(vertices), len(vertices)))

        # Compute the distance matrix
        for i in range(len(vertices)):
            for j in range(len(vertices)):
                if i == j:
                    continue
                u = vertices[i]
                v = vertices[j]
                weight = self.G.get_edge_data(u=u, v=v)
                if weight is None:
                    weight = 0
                else:
                    weight = weight["weight"]
                distance_matrix[i][j] = weight
                distance_matrix[j][i] = weight

        # Use MDS to get the positions of the vertices in a lower-dimensional space
        mds = MDS(dissimilarity="precomputed", n_components=distance_matrix.shape[0])
        positions = mds.fit_transform(distance_matrix)

        # Compute the centroid as the mean position of all vertices
        centroid = np.mean(positions, axis=0)

        # Compute the Euclidean distance from each vertex to the centroid
        distances = np.linalg.norm(positions - centroid, axis=1)

        # Find the index of the vertex closest to the centroid and return it
        centroid_index = np.argmin(distances)
        return vertices[centroid_index]

    def compute_all_centroids(self):
        """Computes centroids for all clusters and returns them in a dictionary."""
        clusters_list = self.clusters["cluster_id"].unique().tolist()
        cluster_to_centroid = {cluster: self.compute_cluster_centroid(cluster) for cluster in clusters_list}
        return cluster_to_centroid

    def save_centroids(self, cluster_to_centroid, output_path):
        """Saves the computed cluster centroids to a pickle file."""
        with open(output_path, "wb") as f:
            pickle.dump(obj=cluster_to_centroid, file=f, protocol=-1)

    def run(self, output_path):
        """Runs the centroid computation and saves the result to a file."""
        cluster_to_centroid = self.compute_all_centroids()
        self.save_centroids(cluster_to_centroid, output_path)


# Main execution
def main():
    clustering_algorithm = "Louvain"
    clusters_path = f"male_clusters_{clustering_algorithm}.csv"
    graph_path = 'male_graph.edgelist'
    output_path = f"cluster_to_centroid_{clustering_algorithm}"

    # Read the graph
    G = nx.read_weighted_edgelist(graph_path)

    # Initialize the cluster centroid calculator
    centroid_calculator = ClusterCentroidCalculator(clusters_path, G, clustering_algorithm=clustering_algorithm)

    # Run the computation and save results
    centroid_calculator.run(output_path)


if __name__ == "__main__":
    main()
