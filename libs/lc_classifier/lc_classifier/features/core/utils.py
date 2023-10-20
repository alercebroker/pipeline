import numba
import numpy as np


@numba.jit(nopython=True)
def is_sorted(a):
    for i in range(a.size-1):
        if a[i+1] < a[i]:
            return False
    return True


def mag_to_flux(mag: np.ndarray):
    """Converts a list of magnitudes into flux."""
    return 10 ** (-(mag + 48.6) / 2.5 + 26.0)
