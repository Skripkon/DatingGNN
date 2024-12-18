from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import community as community_louvain
import igraph as ig
import leidenalg as la
import networkx as nx
import pandas as pd
import pickle
import torch
import torch_geometric
from tqdm import tqdm


class CommunityDetection:
    def __init__(self, graph_path, output_path, data=None):
        self.graph_path = graph_path
        self.output_path = output_path
        # Read graph
        self.G = nx.read_weighted_edgelist(graph_path)
        self.data = data

    def apply_leiden(self):
        ig_graph = ig.Graph.TupleList(self.G.edges(data=True), weights=True)

        partition = la.find_partition(ig_graph, la.ModularityVertexPartition)

        return {node: cluster for node, cluster in zip(self.G.nodes(), partition.membership)}

    def apply_louvain(self):
        partition = community_louvain.best_partition(self.G)
        return partition

    def apply_gcn(self):
        gcn_model = GCN(input_dim=self.data.x.shape[1], hidden_dim=16, output_dim=8)
        gcn_embeddings = self.train_model(gcn_model)
        kmeans = KMeans(n_clusters=25)
        kmeans.fit(gcn_embeddings)
        gcn_labels = kmeans.labels_
        return gcn_labels

    def apply_graphsage(self):
        graphsage_model = GraphSAGE(input_dim=self.data.x.shape[1], hidden_dim=16, output_dim=8)
        graphsage_embeddings = self.train_model(graphsage_model)
        kmeans = KMeans(n_clusters=25)
        kmeans.fit(graphsage_embeddings)
        graphsage_labels = kmeans.labels_
        return graphsage_labels

    def train_model(self, model, epochs=1000, lr=0.01):
        """Train the GCN or GraphSAGE model."""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = model(self.data)
            loss = torch.nn.functional.mse_loss(embeddings, embeddings)
            loss.backward()
            optimizer.step()
        return embeddings.detach().numpy()

    def save_clusters(self, partition, method_name):
        """Saves the community clusters to a CSV file."""
        df = pd.DataFrame({
            'vertex_id': list(partition.keys()),
            'cluster_id': list(partition.values())
        })
        df.to_csv(f'{self.output_path}_{method_name}.csv', index=False)

    def run(self):
        leiden_partition = self.apply_leiden()
        self.save_clusters(leiden_partition, 'Leiden')

        louvain_partition = self.apply_louvain()
        self.save_clusters(louvain_partition, 'Louvain')

        gcn_labels = self.apply_gcn()
        gcn_partition = {i: label for i, label in enumerate(gcn_labels)}
        self.save_clusters(gcn_partition, 'GCN')

        graphsage_labels = self.apply_graphsage()
        graphsage_partition = {i: label for i, label in enumerate(graphsage_labels)}
        self.save_clusters(graphsage_partition, 'GraphSAGE')


# Main execution
def main():
    graph_path = 'male_graph.edgelist'
    output_path = 'male_clusters'
    data = torch_geometric.data.Data()

    community_detector = CommunityDetection(graph_path, output_path, data)

    community_detector.run()


if __name__ == "__main__":
    main()
