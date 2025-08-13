import numpy as np
import io
from astropy.io import fits


def extract_image_from_fits(fits_file: bytes) -> np.ndarray:
    with fits.open(io.BytesIO(fits_file)) as hdul:
        stamp = hdul[0].data
        # variance_uncertainty = hdul[1].data
        # psf_image = hdul[2].data

    return stamp
