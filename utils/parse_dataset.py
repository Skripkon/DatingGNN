import pandas as pd


def parse_dataset() -> pd.DataFrame:
    """
    Parses the dataset from a CSV file and returns it as a Pandas DataFrame.

    This function loads a CSV file (`data/okcupid_profiles.csv`) that contains user profiles
    from an online dating platform (OkCupid). The data is read into a Pandas DataFrame for further processing.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed dataset. The dataset consists of user profiles, 
        with various attributes (e.g., gender, age, profile text, etc.) as columns.

    Example:
        data = parse_dataset()
        print(data.head())

    Raises:
        FileNotFoundError: If the CSV file (`data/okcupid_profiles.csv`) is not found in the expected location.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    # Read the dataset from the specified CSV file path
    data = pd.read_csv("data/okcupid_profiles.csv")
    return data

# TODO


def preprocess_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the input dataset and returns a cleaned version.

    This function is a placeholder and is meant to be implemented for cleaning, transforming, 
    or processing the dataset in any required way. The preprocessing might include steps like:
    - Handling missing data ???
    - Encoding categorical variables ???
    - Normalizing or scaling numerical features ???
    - Text preprocessing (e.g., tokenization, removing stopwords, etc.) ???
    - Feature engineering ???

    Args:
        data (pd.DataFrame): The raw dataset to be preprocessed, which is expected to be a DataFrame.

    Returns:
        pd.DataFrame: A cleaned version of the dataset after preprocessing. The structure of the returned
        DataFrame will depend on the specific preprocessing applied.

    Raises:
        NotImplementedError: This function is currently not implemented. It should raise an exception if
        called before being implemented.

    Example:
        data = parse_dataset()
        preprocessed_data = preprocess_dataset(data)
        print(preprocessed_data.head())
    """
    raise NotImplementedError("The preprocess_dataset function is not yet implemented.")
