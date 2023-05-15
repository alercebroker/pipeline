import pandas as pd
from mhps import statistics, flux_statistics


def mhps(df: pd.DataFrame, t1: float, t2: float, flux: bool = False) -> pd.Series:
    """Mexican hat power spectrum features.

    Args:
        df: Frame with magnitudes, errors and times. Expects a single object and band at a time
        t1: Low frequency timescale (in days)
        t2: High frequency timescale (in days)
        flux: Whether the input comes in flux, rather than magnitude. Field names do not change

    Returns:
        pd.Series: Power spectrum features
    """
    mag, e_mag, mjd = df[["mag_ml", "e_mag_ml", "mjd"]].T.values

    r, l, h, nz, f = flux_statistics(mag, e_mag, mjd, t1, t2) if flux else statistics(mag, e_mag, mjd, t1, t2)
    return pd.Series({"MHPS_ratio": r, "MHPS_low": l, "MHPS_high": h, "MHPS_non_zero": nz, "MHPS_PN_flag": f})

