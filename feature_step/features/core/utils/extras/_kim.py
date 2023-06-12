import numpy as np
import pandas as pd

from ._utils import reformat, empty

INDICES = ["Psi_CS", "Psi_eta"]


def kim(df: pd.DataFrame, period: float) -> pd.Series:
    r"""Compute phase-folded light-curve parameters.

    Computes the cumulative sum and :math:`\eta_e`.

    Args:
        df: Frame with magnitudes and times. Expects a single object and band at a time
        period: Multi-band period

    Returns:
        pd.Series: Phase-folded light-curve parameters
    """
    if len(df) <= 2:
        return pd.Series([np.nan, np.nan], index=INDICES)
    mag, mjd = df[["mag_ml", "mjd"]].T.values

    mag = mag[np.argsort((mjd % (2 * period)) / (2 * period))]
    sigma = np.std(mag)
    s = np.cumsum(mag - np.mean(mag)) / (mag.size * sigma)
    psi_cs = np.max(s) - np.min(s)
    psi_eta = np.sum(np.diff(mag) ** 2) / ((mag.size - 1) * sigma ** 2)

    return pd.Series([psi_cs, psi_eta], index=INDICES)


def apply_kim(df: pd.DataFrame, period: float, fids: tuple[str, ...]) -> pd.Series:
    r"""Compute phase-folded light-curve parameters.

    Computes the cumulative sum and :math:`\eta_e`.

    Args:
        df: Frame with magnitudes and times. Expects a single object
        period: Multi-band period
        fids: Bands required. If not present, it will fill features with NaN

    Returns:
        pd.Series: Phase-folded light-curve parameters
    """
    return reformat(df.groupby("fid").apply(kim, period=period), fids)


def empty_kim(fids: tuple[str, ...]) -> pd.Series:
    """Generates a series of phase-folded features filled with NaN. This is a convenience function.

    Args:
        fids: Bands required

    Returns:
        pd.Series: Phase-folded light-curve parameters, all filled with NaN
    """
    return empty(INDICES, fids)
