import functools

import numpy as np
import pandas as pd

from ._utils import empty, multiindex


@functools.lru_cache()
def _indices(factors: tuple[str, ...]) -> list[str]:
    """Power rate indices used in output pandas series"""
    return [f"Power_rate_{factor}" for factor in factors]


@functools.lru_cache()
def _as_float(factors: tuple[str, ...]) -> list[float]:
    """Transforms factors to floats"""
    return [
        functools.reduce(
            lambda x, y: x.__truediv__(y), (float(d) for d in f.split("/"))
        )
        for f in factors
    ]


def power_rates(
    freq: np.ndarray, per: np.ndarray, factors: tuple[str, ...]
) -> pd.Series:
    imax = np.argmax(per)
    max_power, max_freq = per[imax], freq[imax]
    argsort = np.argsort(freq)
    freq, per = freq[argsort], per[argsort]

    per_factor = np.full(len(factors), np.nan)
    for i, factor in enumerate(_as_float(factors)):
        per_factor[i] = _power_rate(freq, per, max_freq, max_power, factor)

    return pd.Series(per_factor, index=_indices(factors))


def _power_rate(
    freq: np.ndarray, per: np.ndarray, frequency: float, power: float, factor: float
) -> pd.Series:
    desired = frequency / factor

    i = np.searchsorted(freq, desired)
    if i == 0 or i == freq.size:
        return per[i if i == 0 else i - 1] / power
    mean = np.mean(freq[i - 1 : i + 1])
    return per[i if desired > mean else i - 1] / power


def apply_power_rates(
    freq: np.ndarray, per: np.ndarray, factors: tuple[str, ...], fids: tuple[str, ...]
) -> pd.Series:
    return power_rates(freq, per, factors).set_axis(
        multiindex(tuple(_indices(factors)), ("".join(fids),))
    )


def empty_power_rates(factors: tuple[str, ...], fids: tuple[str, ...]) -> pd.Series:
    return empty(_indices(factors), ("".join(fids),))
