import community as community_louvain
import igraph as ig
import leidenalg as la
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import SpectralClustering
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
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

        # Apply Leiden algorithm
        partition = la.find_partition(ig_graph, la.ModularityVertexPartition, n_iterations=10)

        # Calculate and print modularity for Leiden
        modularity = partition.modularity
        print(f"Leiden Modularity: {modularity:.4f}")
        print(f"Leiden clusters: {len(set(partition.membership))}")
        return {node: cluster for node, cluster in zip(self.G.nodes(), partition.membership)}

    def apply_louvain(self):
        # Apply Louvain algorithm
        partition = community_louvain.best_partition(self.G)

        # Calculate and print modularity for Louvain
        modularity = community_louvain.modularity(partition, self.G)
        print(f"Louvain Modularity: {modularity:.4f}")
        print(f"Lovain clusters: {len(set(partition.values()))}")
        return partition

    def apply_node2vec_kmeans(self):
        # Apply Node2Vec
        node2vec = Node2Vec(self.G, dimensions=64, walk_length=80, num_walks=200, workers=8, quiet=False)
        model = node2vec.fit()

        # Generate embeddings
        embeddings = [model.wv[str(node)] for node in self.G.nodes()]

        best_modularity = float('-inf')
        best_labels = None

        with tqdm(total=99, desc="Searching for the optimal number of clusters") as pbar:
            for n_clusters in range(2, 101):
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(embeddings)
                labels = kmeans.labels_

                # Prepare clusters for modularity calculation
                communities = {i: [] for i in range(n_clusters)}
                for node, label in zip(self.G.nodes(), labels):
                    communities[label].append(node)

                community_list = list(communities.values())

                # Calculate modularity
                modularity = nx.algorithms.community.modularity(self.G, community_list)
                if modularity > best_modularity:
                    best_modularity = modularity
                    best_labels = labels

                # Update progress bar with the latest modularity value
                pbar.set_postfix({"Modularity": modularity})
                pbar.update(1)
                n_clusters += 1

        print(f"Node2Vec + KMeans Modularity: {best_modularity:.4f}")
        print(f"Node2Vec clusters: {len(set(best_labels))}")

        return {node: label for node, label in zip(self.G.nodes(), best_labels)}

    def apply_graphsage_kmeans(self):
        # Relabel nodes for indexing
        mapping = {node: i for i, node in enumerate(self.G.nodes())}
        inverse_mapping = {i: node for i, node in enumerate(self.G.nodes())}
        G = nx.relabel_nodes(self.G, mapping)
        # Convert graph to PyTorch Geometric format
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        num_nodes = G.number_of_nodes()

        N_DIMS = 128
        x = torch.rand((num_nodes, N_DIMS), dtype=torch.float)  # Random node features

        data = Data(x=x, edge_index=edge_index)

        class GraphSAGE(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super(GraphSAGE, self).__init__()
                self.conv1 = SAGEConv(in_channels, hidden_channels)
                self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, hidden_channels)
                self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
                self.conv3 = SAGEConv(hidden_channels, hidden_channels)
                self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
                self.conv4 = SAGEConv(hidden_channels, out_channels)

            def forward(self, x, edge_index):
                x = F.relu(self.bn1(self.conv1(x, edge_index)))
                x = F.relu(self.bn2(self.conv2(x, edge_index)))
                x = F.relu(self.bn3(self.conv3(x, edge_index)))
                return self.conv4(x, edge_index)

        current_lr = 0.001
        model = GraphSAGE(in_channels=N_DIMS, hidden_channels=384, out_channels=N_DIMS)
        optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)

        # Train model
        for _ in range(61):
            model.train()
            optimizer.zero_grad()
            z = model(data.x, data.edge_index)
            loss = F.mse_loss(z, data.x)
            loss.backward()
            optimizer.step()
            if _ % 15 == 0:
                print("loss", loss.item())
                current_lr /= 10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

        # Get embeddings
        model.eval()
        embeddings = model(data.x, data.edge_index).detach().numpy()
        G = nx.relabel_nodes(self.G, inverse_mapping)

        best_modularity = float('-inf')
        best_labels = None

        with tqdm(total=99, desc="Searching for the optimal number of clusters") as pbar:
            for n_clusters in range(2, 101):
                kmeans = KMeans(n_clusters=n_clusters)
                labels = kmeans.fit_predict(embeddings)
                # Prepare clusters for modularity calculation
                communities = {i: [] for i in range(n_clusters)}
                for node, label in zip(G.nodes(), labels):
                    communities[label].append(node)

                community_list = list(communities.values())

                # Calculate modularity
                modularity = nx.algorithms.community.modularity(G, community_list)
                if modularity > best_modularity:
                    best_modularity = modularity
                    best_labels = labels

                # Update progress bar with the latest modularity value
                pbar.set_postfix({"Modularity": modularity})
                pbar.update(1)
                n_clusters += 1

        print(f"GraphSAGE + KMeans: {best_modularity:.4f}")
        print(f"GraphSAGE + KMeans clusters: {len(set(best_labels))}")

        return {node: label for node, label in zip(G.nodes(), labels)}

    def save_clusters(self, partition, method_name):
        """Saves the community clusters to a CSV file."""
        df = pd.DataFrame({
            'vertex_id': list(partition.keys()),
            'cluster_id': list(partition.values())
        })
        df.to_csv(f'{self.output_path}_{method_name}.csv', index=False)

    def run(self):
        # print("Running Leiden...")
        # leiden_partition = self.apply_leiden()
        # self.save_clusters(leiden_partition, 'Leiden')

        # print("Running Louvain...")
        # louvain_partition = self.apply_louvain()
        # self.save_clusters(louvain_partition, 'Louvain')

        print("Running Node2Vec + KMeans...")
        node2vec_partition = self.apply_node2vec_kmeans()
        self.save_clusters(node2vec_partition, 'Node2Vec_KMeans')

        # print("Running GraphSAGE + KMeans...")
        # graphsage_partition = self.apply_graphsage_kmeans()
        # self.save_clusters(graphsage_partition, 'GraphSAGE_KMeans')


# Main execution
def main():
    graph_path = 'male_graph.edgelist'
    output_path = 'male_clusters'

    data = torch_geometric.data.Data()
    community_detector = CommunityDetection(graph_path, output_path, data)
    community_detector.run()


if __name__ == "__main__":
    main()
