import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import pickle
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

WEIGHT_THRESHOLD = 0.88
MIN_COMPONENT_SIZE = 20
WEIGHTS = np.array([0.4, 0.1, 0.05, 0.35, 0.1])  # categorical features, age, height, essays, location


class MaleGraph:
    def __init__(self, data_path, features_path, embeddings_path, weight_threshold=WEIGHT_THRESHOLD):
        self.data_path = data_path
        self.features_path = features_path
        self.embeddings_path = embeddings_path
        self.weight_threshold = weight_threshold

        # Load data
        self.data: pd.DataFrame = pd.read_csv(data_path)
        GRAPH_SIZE = 25_000  # TO BE INCREASED
        self.male_indexes = self.data[self.data["sex"] == "m"].index.values[:GRAPH_SIZE]
        self.data = self.data.iloc[self.male_indexes]

        self.cat_features = ["body_type", "drinks", "education", "smokes", "likes_dogs", "likes_cats", "religion", "job"]
        self.categorical_data: pd.DataFrame = self.data[self.cat_features]
        self.categorical_data = self.categorical_data.apply(LabelEncoder().fit_transform)

        self.encoded_features = self.categorical_data.values
        self.distances: pd.DataFrame = pd.read_csv("../data/distances.csv", index_col=0)
        self.distances = (self.distances - self.distances.min().min()) / (self.distances.max().max() - self.distances.min().min())
        self.distances = self.distances.to_dict()
        self.cities = self.data["location"].values

        with open(embeddings_path, "rb") as f:
            self.embeddings_male = pickle.load(f)[0:GRAPH_SIZE]
        self.male_idx_to_absolute_idx = {male_idx: abs_idx for male_idx, abs_idx in zip(self.male_indexes, np.arange(len(self.male_indexes)))}

        print("Computing nominal features similiaritiess")
        self.cat_feature_similarities = self.compute_cat_feature_similarities()
        print("Computing age similiaritiess")
        self.age_similarities = self.compute_age_similarities()
        print("Computing height similiaritiess")
        self.height_similarities = self.compute_height_similarities()
        print("Computing essays similarities")
        self.essay_similarities = self.compute_essay_similarities()
        print("Computing total similiarities")
        self.similiarities = self.compute_total_similarities()

    def compute_cat_feature_similarities(self):
        """Precomputes the distance metric of the categorical features for all males"""
        # Compute the Hamming distance matrix
        hamming_dist = cdist(self.encoded_features, self.encoded_features, metric='hamming')
        return 1 - hamming_dist  # convert distances to similiarities

    def compute_age_similarities(self):
        """Precomputes age-based similarity."""
        age_diff = np.abs(self.data["age"].values[:, np.newaxis] - self.data["age"].values)
        age_range = self.data["age"].max() - self.data["age"].min()
        normalized_age_diff = age_diff / age_range if age_range != 0 else age_diff
        return 1 - normalized_age_diff

    def compute_height_similarities(self):
        """Precomputes height-based similarity."""
        height_diff = np.abs(self.data["height"].values[:, np.newaxis] - self.data["height"].values)
        height_range = self.data["height"].max() - self.data["height"].min()
        normalized_height_diff = height_diff / height_range if height_range != 0 else height_diff
        return 1 - normalized_height_diff

    def compute_essay_similarities(self):
        return (cosine_similarity(self.embeddings_male, self.embeddings_male) + 1) / 2

    def compute_total_similarities(self):
        """Precomputes total similarities between all male pairs using vectorized operations."""

        # Stack all similarity matrices along a new axis and calculate the weighted sum
        total_similarities = (
            WEIGHTS[0] * self.cat_feature_similarities +
            WEIGHTS[1] * self.age_similarities +
            WEIGHTS[2] * self.height_similarities +
            WEIGHTS[3] * self.essay_similarities
        )

        return total_similarities

    def build_graph(self):
        """Builds the graph by adding nodes and fully connecting nodes with the same categorical features."""
        G = nx.Graph()

        # Add male nodes to the graph
        for male_idx in self.male_indexes:
            G.add_node(male_idx)

        # Add edges
        for m1 in tqdm(self.male_indexes, desc="Building edges"):
            for m2 in self.male_indexes:
                if m1 == m2:
                    continue

                m1_idx = self.male_idx_to_absolute_idx[m1]
                m2_idx = self.male_idx_to_absolute_idx[m2]

                total_sim = self.similiarities[m1_idx][m2_idx] + WEIGHTS[4] * self.distances[self.cities[m1_idx]][self.cities[m2_idx]]
                if total_sim >= self.weight_threshold:
                    G.add_edge(m1, m2)

        return G

    def remove_small_components(self, G, N):
        """
        Remove all nodes from components of size <= N from the graph G.

        Parameters:
            G (networkx.Graph): The input graph.
            N (int): The maximum size of components to remove.

        Returns:
            networkx.Graph: The modified graph with small components removed.
        """
        # Identify all connected components
        components = nx.connected_components(G)

        # Find components whose size is <= N
        nodes_to_remove = [node for component in components if len(component) <= N for node in component]

        # Remove these nodes from the graph
        G.remove_nodes_from(nodes_to_remove)

        return G

    def save_graph(self, graph, output_path):
        graph = self.remove_small_components(graph, MIN_COMPONENT_SIZE)
        print(f"Number of nodes: {len(graph.nodes())}")
        print(f"number of edges: {len(graph.edges())}")
        """Saves the graph to a file."""
        nx.write_edgelist(graph, path=output_path, data=[])


def main():
    data_path = "../data/preprocessed_data.csv"
    features_path = "../data/features"
    embeddings_path = "../embedders/embeddings/embeddings_male.obj"
    output_graph_path = "male_graph.edgelist"

    # Initialize the graph builder
    graph_builder = MaleGraph(data_path, features_path, embeddings_path)

    # Build the graph
    graph = graph_builder.build_graph()

    # Save the graph to a file
    graph_builder.save_graph(graph, output_graph_path)


if __name__ == "__main__":
    main()
