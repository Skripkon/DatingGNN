import community as community_louvain
import igraph as ig
import leidenalg as la
import networkx as nx
import pandas as pd
import torch_geometric


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

        return {node: cluster for node, cluster in zip(self.G.nodes(), partition.membership)}

    def apply_louvain(self):
        # Apply Louvain algorithm
        partition = community_louvain.best_partition(self.G)

        # Calculate and print modularity for Louvain
        modularity = community_louvain.modularity(partition, self.G)
        print(f"Louvain Modularity: {modularity:.4f}")

        return partition

    def save_clusters(self, partition, method_name):
        """Saves the community clusters to a CSV file."""
        df = pd.DataFrame({
            'vertex_id': list(partition.keys()),
            'cluster_id': list(partition.values())
        })
        df.to_csv(f'{self.output_path}_{method_name}.csv', index=False)

    def run(self):
        print("Running Leiden...")
        leiden_partition = self.apply_leiden()
        self.save_clusters(leiden_partition, 'Leiden')

        print("Running Louvain...")
        louvain_partition = self.apply_louvain()
        self.save_clusters(louvain_partition, 'Louvain')


# Main execution
def main():
    graph_path = 'male_graph.edgelist'
    output_path = 'male_clusters'
    data = torch_geometric.data.Data()

    community_detector = CommunityDetection(graph_path, output_path, data)

    community_detector.run()


if __name__ == "__main__":
    main()
