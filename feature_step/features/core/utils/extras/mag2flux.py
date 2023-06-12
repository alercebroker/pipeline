import numpy as np


def ztf(mag: np.ndarray | float) -> np.ndarray | float:
    """Magnitude to flux conversion for ZTF.

    Args:
        mag: Magnitudes

    Returns:
        np.ndarray | float: Fluxes
    """
    return 10 ** (-(mag + 48.6) / 2.5 + 26.0)
