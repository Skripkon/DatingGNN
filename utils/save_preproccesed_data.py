import pandas as pd

from parse_dataset import parse_dataset, preprocess_dataset


if __name__ == "__main__":
    data: pd.DataFrame = parse_dataset()
    data = preprocess_dataset(data)
    data.to_csv("data/preprocessed_data.csv", index=False)
