import numpy as np

def calculate_stats_coordinates(coordinates: list, e_coordinates: list):
    coordinates, e_coordinates = np.array(coordinates), np.array(e_coordinates)
    e_coordinates = e_coordinates / 3600
    num_coordinate = np.sum(coordinates / e_coordinates**2)
    den_coordinate = np.sum(1 / e_coordinates**2)
    mean_coordinate = num_coordinate / den_coordinate
    e_coord = np.sqrt(1 / den_coordinate) * 3600
    return mean_coordinate, e_coord