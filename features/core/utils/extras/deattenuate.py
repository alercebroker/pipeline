import extinction
import numpy as np
from astropy.cosmology import WMAP5


def _distmod(z):
    return WMAP5.distmod(z).to("mag").value


def lsst(
    flux: np.ndarray, error: np.ndarray, band: np.ndarray, mwebv: float, zhost: float
):
    """Modifies flux and error inplace"""
    zhost = zhost if zhost >= 0.003 else 0
    z_deatt = 10 ** (-(_distmod(0.3) - _distmod(zhost)) / 2.5) if zhost else 1

    rv = 3.1
    av = rv * mwebv

    wavelengths = {  # ¿Central? Wavelength (in Angstroms) for each band
        "u": 3671.0,
        "g": 4827.0,
        "r": 6223.0,
        "i": 7546.0,
        "z": 8691.0,
        "Y": 9712.0,
    }

    for fid in np.unique(band):
        mask = band == fid
        (dust_deatt,) = 10 ** (
            extinction.odonnell94(np.full(1, wavelengths[fid]), av, rv) / 2.5
        )

        flux[mask] *= z_deatt * dust_deatt
        error[mask] *= z_deatt * dust_deatt

    return flux, error
