import pandas as pd
import networkx as nx
from typing import List
from tqdm import tqdm


def build_graph(data: pd.DataFrame, pairwise_features: List[str] = None) -> nx.Graph:
    """
    Builds a graph of users based on compatibility defined by shared features and sex/orientation conditions.

    Parameters:
    - data (pd.DataFrame): DataFrame where each row represents a user and columns include user features.
    - pairwise_features (List[str], optional): List of feature names to compare for compatibility. 
      Default is ['status', 'drinks', 'drugs', 'smokes', 'pets', 'offspring'].

    Returns:
    - nx.Graph: A NetworkX graph object where nodes represent users and edges represent compatibility.
    """

    # Default features to compare if not provided
    if pairwise_features is None:
        pairwise_features = ['status', 'drinks', 'drugs', 'smokes', 'pets', 'offspring']

    def check_compatibility(user1: pd.Series, user2: pd.Series) -> bool:
        """
        Checks if two users are compatible based on their sex, orientation, and shared features.

        Parameters:
        - user1, user2 (pd.Series): Two user rows from the DataFrame to be compared.

        Returns:
        - bool: True if users are compatible, otherwise False.
        """
        # Step 1: Check the sex and orientation compatibility
        if not is_sex_orientation_compatible(user1, user2):
            return False

        # Step 2: Check if at least 4 out of 6 features match
        match_count = sum(user1[feature] == user2[feature] for feature in pairwise_features)
        return match_count >= 5

    def is_sex_orientation_compatible(user1: pd.Series, user2: pd.Series) -> bool:
        """
        Check if two users are compatible based on sex and orientation.

        Compatibility conditions:
        - Opposite sex and both straight/bisexual
        - Same sex and both gay/bisexual

        Parameters:
        - user1, user2 (pd.Series): Two user rows to be compared.

        Returns:
        - bool: True if users are compatible based on sex and orientation, otherwise False.
        """
        sex1, sex2 = user1['sex'], user2['sex']
        orientation1, orientation2 = user1['orientation'], user2['orientation']

        if (sex1 != sex2 and orientation1 in ['straight', 'bisexual'] and orientation2 in ['straight', 'bisexual']) or \
           (sex1 == sex2 and orientation1 in ['gay', 'bisexual'] and orientation2 in ['gay', 'bisexual']):
            return True
        return False

    # Step 3: Construct the graph using NetworkX
    G = nx.Graph()

    # Add nodes (each user is a node)
    for user_id in data.index:
        G.add_node(user_id, data=data.loc[user_id])

    # Step 4: Compare every pair of users and add an edge if they are compatible
    for i, user_id1 in tqdm(enumerate(data.index), total=len(data.index), desc="Comparing users"):
        for user_id2 in data.index[i + 1:]:  # Only check pairs once (i < j)
            user1 = data.loc[user_id1]
            user2 = data.loc[user_id2]

            if check_compatibility(user1, user2):
                G.add_edge(user_id1, user_id2)

    return G
