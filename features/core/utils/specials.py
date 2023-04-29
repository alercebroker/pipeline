import functools
import numpy as np

import mhps
import pandas as pd
from turbofats import FeatureSpace
from P4J import MultiBandPeriodogram

from .extras.models import fit_sn_model

FACTORS = ["1/4", "1/3", "1/2", "2", "3", "4"]


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


def periods4apply(
    df: pd.DataFrame, fids: tuple[str, ...], kim: bool = False, power: bool = False, harmonics: int = 0
) -> pd.Series:
    periodogram = _get_multiband_periodogram()

    df = df.groupby("fid").filter(lambda x: len(x) > 5).sort_values("mjd")

    fid = df["fid"].values
    mag, e_mag, mjd = df[["mag_ml", "e_mag_ml", "mjd"]].T.values
    periodogram.set_data(mjd, mag, e_mag, fid)
    try:
        periodogram.frequency_grid_evaluation(fmin=1e-3, fmax=20.0, fresolution=1e-3)
        periodogram.finetune_best_frequencies(n_local_optima=10, fresolution=1e-4)
    except TypeError:
        output = [pd.Series(np.nan, index=_period_indices(fids))]
        if kim:
            output.append(pd.Series(np.nan, index=pd.MultiIndex.from_product([_kim_indices(), fids])))
        if harmonics:
            output.append(pd.Series(np.nan, index=pd.MultiIndex.from_product([_harmonics_indices(harmonics), fids])))
        if power:
            output.append(pd.Series(np.nan, index=_power_ratio_indices()))
        return pd.Series(np.nan, index=_period_indices(fids))

    (frequency, *_), _ = periodogram.get_best_frequencies()
    period = 1 / frequency

    n_best = 100
    freq, per = periodogram.get_periodogram()
    tops = np.sort(per)[-n_best:] + 1e-2
    tops /= np.sum(tops)
    significance = 1 - np.sum(-tops * np.log(tops)) / np.log(n_best)

    per_band = np.full(2 * len(fids), np.nan)
    for i, band in enumerate(fids):
        try:
            period_band = 1 / periodogram.get_best_frequency(band)
        except KeyError:
            continue
        per_band[i] = period_band
        per_band[i + 2] = abs(period - period_band)

    output = [pd.Series([period, significance, *per_band], index=_period_indices(fids))]
    if kim:
        output.append(_ensure_complete(df.groupby("fid").apply(kim4apply, period=period), fids))
    if harmonics:
        output.append(_ensure_complete(df.groupby("fid").apply(harmonics4apply, n=harmonics, period=period), fids))
    if power:
        output.append(power_ratios(freq, per, fids))

    return pd.concat(output)


def _period_indices(fids):
    n_fids = len(fids)
    level0 = ("Multiband_period", "PPE") + ("Period_band",) * n_fids + ("delta_period",) * n_fids
    level1 = ("".join(fids),) * 2 + fids + fids
    return pd.MultiIndex.from_arrays([level0, level1], names=(None, "fid"))


def kim4apply(df: pd.DataFrame, period: float) -> pd.Series:
    mag, mjd = df[["mag_ml", "mjd"]].T.values

    mag = mag[np.argsort((mjd % (2 * period)) / (2 * period))]
    sigma = np.std(mag)
    s = np.cumsum(mag - np.mean(mag)) / (mag.size * sigma)
    psi_cs = np.max(s) - np.min(s)
    psi_eta = np.sum(np.diff(mag) ** 2) / (mag.size * sigma ** 2)

    return pd.Series([psi_cs, psi_eta], index=_kim_indices())


def _kim_indices() -> list[str]:
    return ["Psi_CS", "Psi_eta"]


def harmonics4apply(df: pd.DataFrame, n: int, period: float) -> pd.Series:
    mag, e_mag, mjd = df[["mag_ml", "e_mag_ml", "mjd"]].T.values

    time_freq = 2 * np.pi * (np.arange(n) + 1) / period * mjd[:, None]
    omega = np.hstack([np.ones((mjd.size, 1)), np.cos(time_freq), np.sin(time_freq)])

    weight = 1 / (e_mag + 1e-2)
    w_a = weight[:, None] * omega
    w_b = mag * weight
    coeffs = np.matmul(np.linalg.pinv(w_a), w_b[:, None]).flatten()
    cos = coeffs[1:n + 1]
    sin = coeffs[n + 1:]
    mod = np.sqrt(cos ** 2 + sin ** 2)
    phi = np.arctan2(sin, cos)

    phi -= phi[0] * (np.arange(n) + 1)
    phi = phi[1:] % (2 * np.pi)

    mse = np.mean((np.dot(omega, coeffs) - mag) ** 2)

    return pd.Series(np.hstack([mod, phi, [mse]]), index=_harmonics_indices(n))


def _harmonics_indices(n: int) -> list[str]:
    index = [f"Harmonics_mag_{i + 1}" for i in range(n)]
    index += [f"Harmonics_phase_{i + 1}" for i in range(1, n)] + ["Harmonics_mse"]
    return index


def power_ratios(freq: np.ndarray, per: np.ndarray, fids: tuple[str] = None):
    imax = np.argmax(per)
    max_power, max_freq = per[imax], freq[imax]
    argsort = np.argsort(freq)
    freq, per = freq[argsort], per[argsort]

    per_factor = np.full(len(FACTORS), np.nan)
    for i, factor in enumerate(FACTORS):
        if any(c not in "0123456789/" for c in factor):
            raise ValueError(f"Unsafe factor: {factor}")
        per_factor[i] = _power_ratio(freq, per, max_freq, max_power, eval(factor))

    if fids:
        index = pd.MultiIndex.from_product([_power_ratio_indices(), ["".join(fids),]], names=(None, "fid"))
        return pd.Series(per_factor, index=index)
    return pd.Series(per_factor, index=_power_ratio_indices())


def _power_ratio(freq: np.ndarray, per: np.ndarray, frequency: float, power: float, factor: float) -> pd.Series:
    desired = frequency / factor

    i = np.searchsorted(freq, desired)
    if i == 0 or i == freq.size:
        return per[i if i == 0 else i - 1] / power
    mean = np.mean(freq[i - 1:i + 1])
    return per[i if desired > mean else i - 1] / power


def _power_ratio_indices() -> list[str]:
    return [f"Power_rate_{f}" for f in FACTORS]


def _ensure_complete(df: pd.DataFrame, fids: tuple[str, ...]) -> pd.Series:
    df = df.stack()
    df.index = df.index.swaplevel()
    if (~np.isin(fids, df.index.get_level_values("fid"))).any():
        missing = np.array(fids)[~np.isin(fids, df.index.get_level_values("fid"))]
        columns = df.index.unique(level=0)
        df = pd.concat([df, pd.Series(np.nan, index=pd.MultiIndex.from_product([columns, missing]))])
    return df
