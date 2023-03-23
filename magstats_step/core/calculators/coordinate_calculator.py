import numpy as np
import pandas as pd


def calculate_weights(e_coordinates: pd.Series):
    return 1 / (e_coordinates / 3600) ** 2


def calculate_mean_coordinate(coordinates: pd.Series, weights: pd.Series):
    return np.average(coordinates, weights=weights)


def calculate_e_coord(weights: pd.Series):
    return np.sqrt(1 / np.sum(weights)) * 3600


def calculate_stats_coordinates(coordinates: pd.Series, e_coordinates: pd.Series):
    """TODO: Doc"""
    # This assumes that the error always comes in arcsecs, which is not always true
    # Also, we're now assuming coordinates always comes in degrees (mostly true)
    weights = calculate_weights(e_coordinates)
    mean_coordinate = calculate_mean_coordinate(coordinates, weights)
    e_coord = calculate_e_coord(weights)

    return mean_coordinate, e_coord
