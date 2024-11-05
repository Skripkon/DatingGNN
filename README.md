# Community Detection and Recommendation System for a Dating App

This project implements a community detection and user recommendation system for a dating app using Graph Neural Networks (GNNs).

## Overview

The system performs the following steps:

1. **Community Detection**: 
   - Identifies communities of users based on shared interests, such as communication, friendship, relationships, and sex.

2. **User Ranking**: 
   - For each user, ranks other users within their respective communities to identify the top N most relevant matches.

3. **Graph Representation**:
   - **Nodes**: Represent individual users.
   - **Edges**: Created between users based on:
     - **Cosine similarity** of their profile embeddings (text descriptions).
     - **Mutual interests** such as age, sex, location, etc.

## How it Works

1. **Profile Embedding**: 
   - Users' profile descriptions are embedded into a vector space. Cosine similarity is used to calculate the distance between users' embeddings.

2. **Interest Matching**: 
   - Additional features (age, sex, location, etc.) are used to enhance the edge creation between users with similar interests.

3. **Graph Neural Networks**:
   - GNNs are applied to detect communities within the user graph and rank the most relevant users for recommendations.

## Installation

To install and run this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Skripkon/DatingGNN
cd DatingGNN
pip install -r requirements.txt
```

## For Developers

1. Install the necessary libraries for development:

```bash
pip install -r requirements_dev.txt
```

2. Before pushing any changes, format your code using ```autopep8```:

```bash
autopep8 --in-place $(git ls-files '*.py' '*.ipynb')
```

This will automatically apply PEP 8 style formatting to all Python (```.py```) and Jupyter Notebook (```.ipynb```) files.