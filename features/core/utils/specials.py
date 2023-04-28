import functools
import numpy as np

import mhps
import pandas as pd
from turbofats import FeatureSpace
from P4J import MultiBandPeriodogram

from .extras.models import fit_sn_model


@functools.lru_cache
def _get_feature_space(features: tuple[str, ...]) -> FeatureSpace:
    return FeatureSpace(features)


@functools.lru_cache
def _get_multiband_periodogram() -> MultiBandPeriodogram:
    return MultiBandPeriodogram(method="MHAOV")


def _mag2flux_ztf(mag: np.ndarray) -> np.ndarray:
    return 10 ** (-(mag + 48.6) / 2.5 + 26.0)


def mhps4apply(df: pd.DataFrame, t1: float, t2: float) -> pd.Series:
    mag, e_mag, mjd = df[["mag_ml", "e_mag_ml", "mjd"]].T.values

    r, l, h, nz, f = mhps.statistics(mag, e_mag, mjd, t1, t2)
    return pd.Series({"MHPS_ratio": r, "MHPS_low": l, "MHPS_high": h, "MHPS_non_zero": nz, "MHPS_PN_flag": f})


def fats4apply(df: pd.DataFrame, features: tuple[str, ...]) -> pd.Series:
    space = _get_feature_space(features)

    df = df.set_index("id")[["mag_ml", "e_mag_ml", "mjd"]]
    df = df.rename(columns={"mag_ml": "magnitude", "e_mag_ml": "error", "mjd": "time"})
    return space.calculate_features(df).squeeze(axis="index")


def sn4apply_ztf(df: pd.DataFrame) -> pd.Series:
    mag, e_mag, mjd = df[["mag", "e_mag", "mjd"]].T.values
    flux = _mag2flux_ztf(mag)
    e_flux = _mag2flux_ztf(mag - e_mag) - flux

    result = fit_sn_model(mjd.astype(np.float32), flux.astype(np.float32), e_flux)
    return result.rename({c: f"SPM_{c}" for c in result.index})


def periods4apply(df: pd.DataFrame, filters: tuple[str, ...]) -> pd.Series:
    periodogram = _get_multiband_periodogram()

    n_filters = len(filters)
    level0 = ("Multiband_period", "PPE") + ("Period_band",) * n_filters + ("delta_period",) * n_filters
    level1 = (("".join(filters)), ("".join(filters))) + filters + filters
    index = pd.MultiIndex.from_arrays([level0, level1], names=(None, "fid"))

    df = df.groupby("fid").filter(lambda x: len(x) > 5).sort_values("mjd")

    fid = df["fid"].values
    mag, e_mag, mjd = df[["mag_ml", "e_mag_ml", "mjd"]].T.values
    periodogram.set_data(mjd, mag, e_mag, fid)
    try:
        periodogram.frequency_grid_evaluation(fmin=1e-3, fmax=20.0, fresolution=1e-3)
        periodogram.finetune_best_frequencies(n_local_optima=10, fresolution=1e-4)
    except TypeError:
        return pd.Series(np.full(2 + 2 * n_filters), index=index)

    (freq, *_), _ = periodogram.get_best_frequencies()
    period = 1 / freq

    n_best = 100
    _, per = periodogram.get_periodogram()
    tops = np.sort(per)[-n_best:] + 1e-2
    tops /= np.sum(tops)
    significance = 1 - np.sum(-tops * np.log(tops)) / np.log(n_best)

    per_band = np.full(2 * n_filters, np.nan)
    for i, band in enumerate(filters):
        try:
            freq_band = 1 / periodogram.get_best_frequency(band)
        except KeyError:
            continue
        per_band[i] = freq_band
        per_band[i + 2] = abs(period - freq_band)

    return pd.Series([period, significance, *per_band], index=index)
