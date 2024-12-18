import leidenalg as la
import igraph as ig
import networkx as nx
import pandas as pd


class CommunityDetection:
    def __init__(self, graph_path, output_path):
        self.graph_path = graph_path
        self.output_path = output_path
        # Read graph
        self.G = nx.read_weighted_edgelist(graph_path)

    def apply_leiden(self):
        """Applies the Leiden algorithm to detect communities in the graph."""
        # Convert NetworkX graph to igraph format
        ig_graph = ig.Graph.TupleList(self.G.edges(data=True), weights=True)
        # Apply Leiden algorithm
        partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
        # Return node to cluster mapping
        return {node: cluster for node, cluster in zip(self.G.nodes(), partition.membership)}

    def save_clusters(self, leiden_partition):
        """Saves the community clusters to a CSV file."""
        df = pd.DataFrame({
            'vertex_id': leiden_partition.keys(),
            'cluster_id': leiden_partition.values()
        })
        df.to_csv(self.output_path, index=False)

    def run(self):
        """Runs the community detection and saves the results."""
        # Apply Leiden community detection
        leiden_partition = self.apply_leiden()
        # Save the results to a CSV file
        self.save_clusters(leiden_partition)


# Main execution
def main():
    graph_path = 'male_graph.edgelist'
    output_path = 'male_clusters.csv'

    # Initialize the community detection object
    community_detector = CommunityDetection(graph_path, output_path)

    # Run community detection and save results
    community_detector.run()


if __name__ == "__main__":
    main()
