import pandas as pd
import networkx as nx
import os
from graph_construct.build_graph import build_graph


def parse_graph(data_path: str = None, test: bool = True) -> nx.Graph:
    """
    Parses or constructs a graph from data stored in a CSV file. If the graph 
    already exists in the form of an edge list, it loads it. Otherwise, it constructs
    the graph from user profile data and saves it as an edge list file.

    Args:
        test (bool): If True, only a subset of the data (first 5000 rows) will be used.
                     Default is True for testing purposes.

    Returns:
        nx.Graph: A NetworkX graph object representing the relationships between users.

    Raises:
        FileNotFoundError: If the data file 'data_path' is missing.
        Exception: If the graph building process fails.
    """

    # Define the path to the graph edge list file
    graph_path = os.path.join("data", "graph.edgelist")

    # Check if the graph already exists as an edge list
    if os.path.isfile(graph_path):
        # Load the graph from the existing edge list file
        G = nx.read_edgelist(graph_path)
    else:
        # If the graph doesn't exist, load the profile data from CSV
        try:
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            raise FileNotFoundError("The required 'okcupid_profiles.csv' file is missing.")

        # In test mode, use a subset of the first 5000 rows
        if test:
            data = data.head(5000)

        # Build the graph using the provided function from the `graph_construct` module
        try:
            G = build_graph(data)
        except Exception as e:
            raise Exception(f"Failed to build the graph: {e}")

        # Save the constructed graph as an edge list for future use
        nx.write_edgelist(G, graph_path)

    return G
