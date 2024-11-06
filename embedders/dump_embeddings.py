import pickle
import torch
import sys
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

from embedders_list import EMBEDDERS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.parse_dataset import parse_dataset


def dump_embeddings(
    data: pd.DataFrame,
    columns_to_embed: list[str],
    pretty_name: str = None,
    hugging_face_name: str = None,
    force_update: bool = False,
) -> None:
    """
    Dumps the embeddings for the specified columns in the DataFrame to a file.

    Args:
        data: The DataFrame containing the text data to be embedded.
        columns_to_embed: A list of column names to embed.
        embedder_name: The name of the SentenceTransformer model to use for embedding.
        force_update: If True, will recompute embeddings even if they already exist.
    """
    # Automatically choose the device (cuda if available, else cpu)
    device = get_optimal_device()
    print(f"Using device: {device}")

    # Define the file path where embeddings will be saved
    embeddings_file = os.path.join("embedders", "embeddings", f"embeddings_{pretty_name}.obj")

    # Check if embeddings already exist and whether to skip computation
    if os.path.isfile(embeddings_file) and not force_update:
        print(f"Embeddings obtained using {pretty_name} are already dumped.")
        return

    # Load the SentenceTransformer model and move it to the specified device
    print(f"Loading model {pretty_name}...")
    embedder = SentenceTransformer(hugging_face_name).to(device)

    # Prepare a dictionary to store embeddings for each column
    embeddings_data = {}

    # Iterate over the columns to embed and generate their embeddings
    for column in columns_to_embed:
        if column not in data.columns:
            print(f"Warning: Column '{column}' not found in the DataFrame. Skipping.")
            continue

        print(f"Embedding column: {column}...")
        column_data = data[column].astype(str).tolist()  # Ensure data is in string format
        column_embeddings = embedder.encode(column_data, convert_to_tensor=True, show_progress_bar=True)

        # Save embeddings in the dictionary
        embeddings_data[column] = column_embeddings

    # Dump the embeddings to the file
    with open(embeddings_file, "wb") as filehandler:
        print(f"Saving embeddings to {embeddings_file}...")
        pickle.dump(embeddings_data, filehandler)

    print(f"Embeddings for {', '.join(columns_to_embed)} saved successfully.")


def get_optimal_device() -> str:
    """
    Returns the optimal device ('cuda' if available, otherwise 'cpu').

    Returns:
        str: The device to use for computation.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Parse the dataset
    data = parse_dataset()

    # Iterate over each embedder and dump embeddings
    for pretty_name, hugging_face_name in EMBEDDERS.items():
        print(f"Processing embeddings for {pretty_name}...")
        dump_embeddings(
            data=data,
            columns_to_embed=["essay0"],  # Ensure this is a list, even if it's a single column
            pretty_name=pretty_name,
            hugging_face_name=hugging_face_name
        )


if __name__ == "__main__":
    main()
