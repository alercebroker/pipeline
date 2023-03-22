import numpy as np
import pandas as pd


def calculate_stats_coordinates(coordinates: pd.Series, e_coordinates: pd.Series):
    """TODO: Doc"""
    # This assumes that the error always comes in arcsecs, which is not always true
    # Also, we're now assuming coordinates always comes in degrees (mostly true)
    e_coordinates = e_coordinates / 3600
    mean_coordinate = np.average(coordinates, weights=e_coordinates**2)
    e_coord = np.sqrt(1 / np.sum(1 / (e_coordinates**2))) * 3600

    return mean_coordinate, e_coord
