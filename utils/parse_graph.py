import networkx as nx
import os


def parse_graph() -> nx.Graph:

    # Define the path to the graph edge list file
    graph_path = os.path.join("final_solution", "male_graph.edgelist")

    # Check if the graph already exists as an edge list
    if os.path.isfile(graph_path):
        # Load the graph from the existing edge list file
        G = nx.read_weighted_edgelist(graph_path)
    else:
        raise "Build graph first! --> $ python3 build_graph.py"

    return G
