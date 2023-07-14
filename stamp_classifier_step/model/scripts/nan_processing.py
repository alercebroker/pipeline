import numpy as np

if __name__ == "__main__":
    x = np.array([[1, 2, 3, 4], [2, 3, np.nan, 5], [np.nan, 5, 2, 3]])
    norm_x = x
    norm_x -= np.nanmin(norm_x)
    norm_x = norm_x / np.nanmax(norm_x)
    norm_x[np.isnan(norm_x)] = 1
