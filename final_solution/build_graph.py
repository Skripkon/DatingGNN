import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm

# Constants
WEIGHT_THRESHOLD = 0.5


class MaleEssayGraph:
    def __init__(self, data_path, features_path, embeddings_path, weight_threshold=WEIGHT_THRESHOLD):
        self.data_path = data_path
        self.features_path = features_path
        self.embeddings_path = embeddings_path
        self.weight_threshold = weight_threshold

        # Load data
        self.data = pd.read_csv(data_path)
        with open(features_path, "rb") as f:
            self.encoded_features = pickle.load(f)

        with open(embeddings_path, "rb") as f:
            self.embeddings_male = pickle.load(f)
        # WARNING: TODO CHANGE 100 to A BIGGER NUMBER !!! WARNING!!!!
        # Select male indexes (first 100)
        self.male_indexes = self.data[self.data["sex"] == "m"].index.values[:100]

    def safe_cosine_similarity(self, embedding1, embedding2):
        """Calculates cosine similarity with error handling for NaN values."""
        if np.isnan(embedding1[0][0]) or np.isnan(embedding2[0][0]):
            return 0
        return cosine_similarity(embedding1, embedding2)[0][0]

    def calculate_edge_weight(self, m1, m2):
        """Calculates edge weight based on essay embeddings and features."""
        total_similarity = 0
        valid_essays = 0

        for j in range(10):  # Iterate over all essays (0 to 9)
            essay = f"essay{j}"
            m1_embedding = self.embeddings_male[essay][m1].reshape(1, -1)
            m2_embedding = self.embeddings_male[essay][m2].reshape(1, -1)
            similarity = self.safe_cosine_similarity(m1_embedding, m2_embedding)
            if similarity > 0:
                total_similarity += similarity
                valid_essays += 1

        edge_weight = total_similarity / valid_essays if valid_essays > 0 else 0
        edge_weight *= 0.3  # Essays account for 30%

        # Add features similarity
        m1_features = self.encoded_features[m1]
        m2_features = self.encoded_features[m2]
        edge_weight += 0.7 * self.safe_cosine_similarity(m1_features.reshape(1, -1), m2_features.reshape(1, -1))

        return edge_weight

    def build_graph(self):
        """Builds the graph by adding nodes and weighted edges."""
        G = nx.Graph()

        # Add male nodes to the graph
        for male_idx in self.male_indexes:
            G.add_node(male_idx)

        # Add edges
        for m1 in tqdm(self.male_indexes, desc="Building edges"):
            for m2 in self.male_indexes:
                if m1 == m2:
                    continue

                edge_weight = self.calculate_edge_weight(m1, m2)

                # Add edge if the weight is above the threshold
                if edge_weight > self.weight_threshold:
                    G.add_edge(m1, m2, weight=edge_weight)

        return G

    def save_graph(self, graph, output_path):
        """Saves the graph to a file."""
        nx.write_edgelist(graph, path=output_path, data=["weight"])


def main():
    data_path = "../data/preprocessed_data.csv"
    features_path = "../data/features"
    embeddings_path = "../embedders/embeddings/embeddings_male.obj"
    output_graph_path = "male_graph.edgelist"

    # Initialize the graph builder
    graph_builder = MaleEssayGraph(data_path, features_path, embeddings_path)

    # Build the graph
    graph = graph_builder.build_graph()

    # Save the graph to a file
    graph_builder.save_graph(graph, output_graph_path)


if __name__ == "__main__":
    main()
