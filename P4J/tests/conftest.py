"""Shared fixtures and synthetic light-curve helpers for the P4J test suite.

All generators are seeded for reproducibility. Light curves are built from a
harmonic series (fundamental + harmonics) plus heteroscedastic Gaussian noise,
sampled at realistic MJD values (~59000-62000) so that the float32 phase-folding
precision behaviour of the kernel is exercised.
"""
import numpy as np
import pytest


def make_lightcurve(
    period,
    n_points=500,
    baseline=(59000.0, 61000.0),
    amplitudes=(1.0, 0.5, 0.25),
    noise=0.02,
    seed=0,
    bands=("g",),
):
    """Build a multi-band periodic light curve.

    Returns (mjd, mag, err, fid) as numpy arrays in the same shape the pipeline
    passes to ``MultiBandPeriodogram.set_data``. All bands share the same
    fundamental period (same physical object observed in several filters), each
    with an independent noise realisation.
    """
    rng = np.random.default_rng(seed)
    f0 = 1.0 / period
    mjd_list, mag_list, err_list, fid_list = [], [], [], []
    for b, band in enumerate(bands):
        t = np.sort(rng.uniform(baseline[0], baseline[1], n_points))
        clean = np.zeros_like(t)
        for k, amp in enumerate(amplitudes):
            clean += amp * np.sin(2.0 * np.pi * t * f0 * (k + 1))
        # small per-band magnitude offset, like different filter zeropoints
        offset = 15.0 + 0.3 * b
        err = np.full(n_points, noise)
        mag = offset + clean + rng.normal(0.0, noise, n_points)
        mjd_list.append(t)
        mag_list.append(mag)
        err_list.append(err)
        fid_list.append(np.full(n_points, band))
    return (
        np.concatenate(mjd_list),
        np.concatenate(mag_list),
        np.concatenate(err_list),
        np.concatenate(fid_list),
    )


# Period/baseline scenarios. The short periods on long baselines are exactly the
# regime where float32 mjd*freq folding loses precision.
PERIOD_CASES = [
    pytest.param(1.0, (59000.0, 61000.0), id="P1.0_2yr"),
    pytest.param(0.2734, (59000.0, 61000.0), id="P0.27_2yr"),
    pytest.param(0.13, (59000.0, 61000.0), id="P0.13_2yr"),
    pytest.param(0.0721, (59000.0, 62650.0), id="P0.07_10yr"),
]


@pytest.fixture
def single_band_lc():
    """A reproducible single-band light curve with a moderate period."""
    period = 0.2734
    mjd, mag, err, fid = make_lightcurve(period, seed=42, bands=("g",))
    return period, mjd, mag, err, fid


@pytest.fixture
def multi_band_lc():
    """A reproducible two-band light curve sharing one period."""
    period = 0.2734
    mjd, mag, err, fid = make_lightcurve(period, seed=7, bands=("g", "r"))
    return period, mjd, mag, err, fid
