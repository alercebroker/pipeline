import numpy as np
import pandas as pd

from ._utils import reformat, empty

INDICES = ["Psi_CS", "Psi_eta"]


def kim(df: pd.DataFrame, period: float) -> pd.Series:
    mag, mjd = df[["mag_ml", "mjd"]].T.values

    mag = mag[np.argsort((mjd % (2 * period)) / (2 * period))]
    sigma = np.std(mag)
    s = np.cumsum(mag - np.mean(mag)) / (mag.size * sigma)
    psi_cs = np.max(s) - np.min(s)
    psi_eta = np.sum(np.diff(mag) ** 2) / (mag.size * sigma ** 2)

    return pd.Series([psi_cs, psi_eta], index=INDICES)


def apply_kim(df: pd.DataFrame, period: float, fids: tuple[str, ...]) -> pd.Series:
    return reformat(df.groupby("fid").apply(kim, period=period), fids)


def empty_kim(fids: tuple[str, ...]) -> pd.Series:
    return empty(INDICES, fids)
