import functools

import numpy as np
import pandas as pd
from celerite2 import GaussianProcess, terms
from scipy import optimize


@functools.lru_cache()
def _get_gp() -> GaussianProcess:
    kernel = terms.RealTerm(a=1, c=10)
    return GaussianProcess(kernel, mean=0)


def gp_drw(df: pd.DataFrame) -> pd.Series:
    mag, err, time = df[["mag_ml", "e_mag_ml", "mjd"]].T.values

    time -= np.min(time)
    mag -= np.mean(mag)
    err = err ** 2

    gp = _get_gp()
    guess = np.zeros(2)

    result = optimize.minimize(
        _nlogp, guess, method="L-BFGS-B", args=(gp, time, mag, err)
    )
    theta0, theta1 = np.exp(result.x)
    return pd.Series([theta0, 1 / theta1], index=["GP_DRW_sigma", "GP_DRW_tau"])


def _set_params(
    params: np.ndarray, gp: GaussianProcess, time: np.ndarray, err: np.ndarray
):
    theta0, theta1 = np.exp(params)

    gp.mean = 0.0
    gp.kernel = terms.RealTerm(a=theta0, c=theta1)
    gp.compute(time, diag=err, quiet=True)


def _nlogp(
    params: np.ndarray,
    gp: GaussianProcess,
    time: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
) -> np.ndarray:
    """Negative log-likelihood"""
    _set_params(params, gp, time, err)
    return -gp.log_likelihood(mag)
