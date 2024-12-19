import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import pickle
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

WEIGHT_THRESHOLD = 0.8


class MaleGraph:
    def __init__(self, data_path, features_path, embeddings_path, weight_threshold=WEIGHT_THRESHOLD):
        self.data_path = data_path
        self.features_path = features_path
        self.embeddings_path = embeddings_path
        self.weight_threshold = weight_threshold

        GRAPH_SIZE = 2000  # TO BE INCREASED
        # Load data
        self.data = pd.read_csv(data_path)
        self.male_indexes = self.data[self.data["sex"] == "m"].index.values[:GRAPH_SIZE]
        self.data = self.data.iloc[self.male_indexes]

        self.cat_features = ["body_type", "drinks", "education", "job", "location", "religion", "smokes", "likes_dogs", "likes_cats"]
        self.categorical_data = self.data[self.cat_features]
        self.categorical_data = self.categorical_data.apply(LabelEncoder().fit_transform)

        self.encoded_features = self.categorical_data.values

        with open(embeddings_path, "rb") as f:
            self.embeddings_male = pickle.load(f)
        self.essay_similiarities = self.compute_essay_similarities()

        self.feature_similiarities = self.compute_feature_similarities()
        self.male_idx_to_absolute_idx = {male_idx: abs_idx for male_idx, abs_idx in zip(self.male_indexes, np.arange(len(self.male_indexes)))}

    def compute_feature_similarities(self):
        """Precomputes the weighted distance metric of features for all males, combining Hamming distance and continuous features."""

        # Compute the Hamming distance matrix
        hamming_dist = cdist(self.encoded_features, self.encoded_features, metric='hamming')

        # Normalize age and height differences
        age_diff = np.abs(self.data["age"].values[:, np.newaxis] - self.data["age"].values)
        height_diff = np.abs(self.data["height"].values[:, np.newaxis] - self.data["height"].values)

        age_range = self.data["age"].max() - self.data["age"].min()
        height_range = self.data["height"].max() - self.data["height"].min()

        normalized_age_diff = age_diff / age_range if age_range != 0 else age_diff
        normalized_height_diff = height_diff / height_range if height_range != 0 else height_diff

        # Combine Hamming distance with normalized age and height differences
        combined_distances = (self.encoded_features.shape[1] * hamming_dist + normalized_age_diff + normalized_height_diff) / (self.encoded_features.shape[1] + 2)
        return combined_distances

    def compute_essay_similarities(self):
        return cosine_similarity(self.embeddings_male, self.embeddings_male)

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

                term1 = self.feature_similiarities[self.male_idx_to_absolute_idx[m1]][self.male_idx_to_absolute_idx[m2]]
                term2 = (self.essay_similiarities[self.male_idx_to_absolute_idx[m1]][self.male_idx_to_absolute_idx[m2]] + 1) / 2
                if 0.8 * term1 + 0.2 * term2 >= self.weight_threshold:
                    G.add_edge(m1, m2)

        return G

    def save_graph(self, graph, output_path):
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
