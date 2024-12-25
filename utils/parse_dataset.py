import pandas as pd
import numpy as np
import pickle
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

    Raises:
        FileNotFoundError: If the CSV file (`data/okcupid_profiles.csv`) is not found in the expected location.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    # Read the dataset from the specified CSV file path
    data = pd.read_csv("data/okcupid_profiles.csv")
    return data


def parse_preprocessed_dataset() -> pd.DataFrame:
    """Similiar to `parse_dataset`, but returns a preprocessed dataset"""

    data = pd.read_csv("data/preprocessed_data.csv")
    return data


def preprocess_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the input dataset and returns a cleaned version.

    Args:
        data (pd.DataFrame): The raw dataset to be preprocessed, which is expected to be a DataFrame.

    Returns:
        pd.DataFrame: A cleaned version of the dataset after preprocessing.

    Example:
        data = parse_dataset()
        preprocessed_data = preprocess_dataset(data)
    """

    def drop_unnecessary_columns(data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop(columns=["income", "diet", "drugs"])  # Too many missing values
        data = data.drop(columns=["last_online", "sign"])  # These column bring no value
        # data = data.drop(columns=[f"essay{i}" for i in range(10)])  # Their embeddings are saved
        data = data.drop(columns=["ethnicity"])  # To avoid being accused of racism
        data = data.drop(columns=["pets"])  # Two other columns were added (likes_cats and likes_dogs)
        data = data.drop(columns=["speaks"])  # 99.9% speak English
        data = data.drop(columns=["offspring"])  # too many missing values
        return data

    # Convert inches to cm
    data["height"] = data["height"].apply(lambda x: round(x * 2.54, 0))

    # Handle missing values
    data["age"] = data["age"].fillna(data["age"].mean())
    data["height"] = data["height"].fillna(data["height"].mean())

    def map_cat(a_string, a_dic):
        """
        a_string: a string we want to map with a_dic
        a_dic: a dictionary whose keys are tuples
        """
        for k in a_dic:
            if a_string in k:
                return a_dic.get(k)
        return np.nan

    body_types_dic = {
        ("skinny", "used up", "thin"): "ectomorph",
        ("average", "fit", "athletic", "jacked"): "mesomorph",
        ("a little extra", "curvy", "full figured"): "endomorph",
    }

    body_categories = CategoricalDtype(
        categories=["ectomorph", "mesomorph", "endomorph"], ordered=True
    )

    data["body_type"] = data.body_type.apply(map_cat, args=(body_types_dic,)).astype(
        body_categories
    )
    data.body_type = data.body_type.fillna(
        body_categories.categories[int(np.median(data.body_type.cat.codes))]
    )

    educationcategories = CategoricalDtype(
        categories=[
            "High school or less",
            "Some college",
            "College or more",
            "Post graduate degree",
        ],
        ordered=True,
    )

    educationdic = {
        (
            "graduated from high school",
            "dropped out of high school",
            "working on high school",
            "high school",
        ): "High school or less",
        (
            "working on two-year college",
            "dropped out of space camp",
            "two-year college",
            "graduated from two-year college",
            "dropped out of college/university",
            "dropped out of two-year college",
            "dropped out of med school",
            "dropped out of law school",
        ): "Some college",
        (
            "working on college/university",
            "working on space camp",
            "graduated from masters program",
            "graduated from college/university",
            "working on masters program",
            "graduated from space camp",
            "college/university",
            "graduated from law school",
            "working on ph.d program",
            "space camp",
            "graduated from med school",
            "working on med school",
            "masters program",
            "dropped out of ph.d program",
            "law school",
            "dropped out of masters program",
            "working on law school",
            "med school",
        ): "College or more",
        ("graduated from ph.d program", "ph.d program"): "Post graduate degree",
    }

    data["education"] = data.education.apply(map_cat, args=(educationdic,)).astype(
        educationcategories
    )

    # Fill in missing values
    data.education = data.education.fillna(
        educationcategories.categories[int(np.median(data["education"].cat.codes))]
    ).astype(educationcategories)

    smokes_dic = {
        ("no",): "no",
        ("sometimes", "when drinking", "trying to quit"): "sometimes",
        ("yes",): "yes",
    }
    smokes_categories = CategoricalDtype(
        categories=["no", "sometimes", "yes"], ordered=True
    )
    data["smokes"] = data.smokes.apply(map_cat, args=(smokes_dic,)).astype(smokes_categories)
    data.smokes = data.smokes.fillna("no").astype(smokes_categories)
    data.smokes.value_counts(dropna=False).sort_index()

    import re

    data.pets = data.pets.fillna("No")

    def likes_pet(s, species):
        dogs_regex = re.compile(r"((?<!dis)likes |has )dogs")
        cats_regex = re.compile(r"((?<!dis)likes |has )cats")
        if species == "dog":
            return "Yes" if bool(dogs_regex.search(s)) else "No"
        elif species == "cat":
            return "Yes" if bool(cats_regex.search(s)) else "No"
        else:
            return "No"

    data["likes_dogs"] = data.pets.apply(likes_pet, args=("dog",))
    data["likes_cats"] = data.pets.apply(likes_pet, args=("cat",))

    data["likes_dogs"] = pd.Categorical(
        data["likes_dogs"], categories=["No", "Yes"], ordered=True
    )
    data["likes_cats"] = pd.Categorical(
        data["likes_cats"], categories=["No", "Yes"], ordered=True
    )

    drinks_dic = {
        ("not at all",): "no",
        ("rarely", "socially"): "sometimes",
        ("often", "very often", "desperately"): "yes",
    }
    drinks_categories = CategoricalDtype(
        categories=["no", "sometimes", "yes"], ordered=True
    )
    data["drinks"] = data.drinks.apply(map_cat, args=(drinks_dic,)).astype(drinks_categories)
    data.drinks = data.drinks.fillna("no").astype(drinks_categories)
    data.drinks.value_counts(dropna=False).sort_index()

    data["job"] = data["job"].fillna("unspecified")  # ~6k out of 50 didn't specify their occupation, but there are onl 21 unique values. Hence, this column might be important
    data = drop_unnecessary_columns(data)

    # drop non-traditional orientation for the sake of simplicity
    data = data[data["orientation"] == "straight"]
    # Expectably, agnosticism is the most popular answer
    data["religion"] = data["religion"].fillna("agnosticism")

    # ========================================================================================== #
    # Computing distances between cities take so much time because there are many
    # cities from which there are only a couple of males.
    # It seems reasonable to precompute distances between all the cities,
    # but for now we will drop all the rare cities from the dataset to speed up the process.

    popular_cities_among_males = data[data["sex"] == "m"]["location"].value_counts().index[0:30]

    def remove_unpopular(s):
        if s not in popular_cities_among_males:
            return popular_cities_among_males[0]
        return s
    data["location"] = data["location"].apply(remove_unpopular)
    #  ========================================================================================== #

    data.to_csv("data/preprocessed_data.csv")
    split_data_by_sex(data=data)
    return data


def split_data_by_sex(data: pd.DataFrame):
    # males_data = data[data["sex"] == "m"]
    # females_data = data[data["sex"] == "f"]
    # # Columns to encode (categorical data)
    # categorical_columns = ['sex', 'orientation', 'body_type', 'drinks', 'education', 'job', 'location', 'religion', 'smokes', 'likes_dogs', 'likes_cats']

    # # One-hot encoding for categorical features
    # encoder = OneHotEncoder(drop='first', sparse_output=False)
    # encoder.fit(data[categorical_columns])

    # males_encoded_categorical = encoder.transform(males_data[categorical_columns])
    # females_encoded_categorical = encoder.transform(females_data[categorical_columns])

    # # Normalize continuous features (age, height)
    # continuous_columns = ['age', 'height']

    # males_scaler = StandardScaler()
    # males_normalized_continuous = males_scaler.fit_transform(males_data[continuous_columns])

    # females_scaler = StandardScaler()
    # females_normalized_continuous = females_scaler.fit_transform(females_data[continuous_columns])

    # # Combine the processed features into a single feature matrix
    # males_features = np.hstack([males_encoded_categorical, males_normalized_continuous])
    # females_features = np.hstack([females_encoded_categorical, females_normalized_continuous])

    # with open("data/males_features", "wb") as f:
    #     pickle.dump(obj=males_features, file=f, protocol=-1)

    # with open("data/females_features", "wb") as f:
    #     pickle.dump(obj=females_features, file=f, protocol=-1)

    # =========================== WITHOUT SPLITTING BY GENDER =======================================
    # Columns to encode (categorical data)
    categorical_columns = ['sex', 'orientation', 'body_type', 'drinks', 'education',
                           'job', 'location', 'religion', 'smokes', 'likes_dogs', 'likes_cats']
    # One-hot encoding for categorical features
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoder.fit(data[categorical_columns])
    encoded_categorical = encoder.transform(data[categorical_columns])

    # Normalize continuous features (age, height)
    continuous_columns = ['age', 'height']
    scaler = StandardScaler()
    normalized_continuous = scaler.fit_transform(data[continuous_columns])

    # Combine the processed features into a single feature matrix
    features = np.hstack([encoded_categorical, normalized_continuous])

    # Save the features to a file
    with open("data/features", "wb") as f:
        pickle.dump(obj=features, file=f, protocol=-1)
    # =================================================================================================
