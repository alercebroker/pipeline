import functools

import numpy as np
import pandas as pd

from ._utils import reformat, empty


@functools.lru_cache()
def _indices(n: int) -> list[str]:
    fmt = "Harmonics_{}"

    def with_i(which, i):
        return f"{fmt.format(which)}_{i}"

    index = [with_i("mag", i + 1) for i in range(n)] + [with_i("phase", i + 1) for i in range(1, n)]
    return index + [fmt.format("mse")]


def harmonics(df: pd.DataFrame, n: int, period: float) -> pd.Series:
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

    return pd.Series(np.hstack([mod, phi, [mse]]), index=_indices(n))


def apply_harmonics(df: pd.DataFrame, n: int, period: float, fids: tuple[str, ...]) -> pd.Series:
    return reformat(df.groupby("fid").apply(harmonics, n=n, period=period), fids)


def empty_harmonics(n: int, fids: tuple[str, ...]) -> pd.Series:
    return empty(_indices(n), fids)
