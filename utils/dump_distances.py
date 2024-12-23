import os
from typing import List, Optional
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from tqdm import tqdm
from parse_dataset import parse_preprocessed_dataset

# Initialize geolocator with a user agent string to comply with
# Nominatim's usage policy
geolocator = Nominatim(user_agent="city_distance_calculator")


def get_coordinates(city: str, n_tries=5) -> Optional[tuple[float, float]]:
    """
    Retrieve the geographic coordinates (latitude, longitude) for a given city using Geopy's Nominatim geocoder.

    Args:
        city (str): The name of the city to geocode.

    Returns:
        Optional[tuple[float, float]]: A tuple containing the latitude and longitude of the city,
                                       or None if the city could not be geocoded.
    """
    while n_tries > 0:
        location = geolocator.geocode(city, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            n_tries -= 1
    return None


def get_distance(
        coordinates1: tuple[float, float], coordinates2: tuple[float, float]) -> float:
    """
    Calculate the geodesic distance (in kilometers) between two geographic coordinates.

    Args:
        coordinates1 (tuple): The (latitude, longitude) of the first location.
        coordinates2 (tuple): The (latitude, longitude) of the second location.

    Returns:
        float: The distance between the two locations in kilometers.
    """
    return geodesic(coordinates1, coordinates2).kilometers


def get_distances_between_cities(
        locations: List[str],
        testing: bool = True) -> pd.DataFrame:
    """
    Create a distance matrix for a list of cities, showing the geodesic distance between each pair of cities.

    Args:
        locations (List[str]): A list of city names.
        testing (bool): If True, only a subset of the locations (first 10) will be processed for testing.

    Returns:
        pd.DataFrame: A Pandas DataFrame with city names as both row and column indices, and the
                      distances between them as values.
    """
    if testing:
        # Limit to first 10 cities for testing purposes
        locations = locations[:10]

    distance_matrix = []

    for city1 in tqdm(locations, desc="Processing cities"):
        coordinates1 = get_coordinates(city1)
        if coordinates1 is None:
            raise (f"Could not find coordinates for {city1}")

        distances = []
        for city2 in locations:
            coordinates2 = get_coordinates(city2)
            if coordinates2 is None:
                raise (f"Could not find coordinates for {city2}")
            else:
                distance = get_distance(coordinates1, coordinates2)
                distances.append(distance)

        distance_matrix.append(distances)

    # Convert the distance matrix into a Pandas DataFrame for easier analysis
    # and manipulation
    return pd.DataFrame(distance_matrix, columns=locations, index=locations)


def save_distance_matrix(distance_matrix: pd.DataFrame, output_path: str):
    """
    Save the distance matrix to a CSV file.

    Args:
        distance_matrix (pd.DataFrame): The distance matrix to be saved.
        output_path (str): The file path where the distance matrix should be saved.
    """
    distance_matrix.to_csv(output_path)
    print(f"INFO:Distance matrix saved to {output_path}")


if __name__ == "__main__":
    # Parse dataset to get city names
    data = parse_preprocessed_dataset()

    # FOR MALES_GRAPH
    locations = data[data["sex"] == "m"]["location"].unique()

    # Generate the distance matrix for the cities
    distance_matrix = get_distances_between_cities(locations, testing=False)

    # Define the path to save the distance matrix as a CSV file
    output_path = "data/distances.csv"

    # Save the distance matrix to a CSV file
    save_distance_matrix(distance_matrix, output_path)
