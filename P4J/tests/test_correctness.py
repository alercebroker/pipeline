"""Correctness tests for the MHAOV MultiBandPeriodogram production path.

These tests pin the public API and the period-recovery behaviour. They must pass
both before and after the performance/precision refactor (no regression), and the
short-period cases are expected to recover *at least as accurately* afterwards.
"""
import numpy as np
import pytest

from P4J import MultiBandPeriodogram
from conftest import make_lightcurve, PERIOD_CASES


def run_pipeline(mjd, mag, err, fid, smallest_period=0.02, largest_period=5.0, shift=0.1):
    """Exact call sequence used by the pipeline's PeriodExtractor."""
    p = MultiBandPeriodogram(method="MHAOV")
    p.set_data(mjd, mag, err, fid)
    p.optimal_frequency_grid_evaluation(
        smallest_period=smallest_period, largest_period=largest_period, shift=shift
    )
    p.optimal_finetune_best_frequencies(times_finer=10.0, n_local_optima=10)
    return p


@pytest.mark.parametrize("period,baseline", PERIOD_CASES)
def test_period_recovery_single_band(period, baseline):
    mjd, mag, err, fid = make_lightcurve(period, baseline=baseline, seed=1, bands=("g",))
    p = run_pipeline(mjd, mag, err, fid, largest_period=2.0 if period < 1 else 5.0)
    best_freq, _ = p.get_best_frequencies()
    recovered = 1.0 / best_freq[0]
    rel_err = abs(recovered - period) / period
    assert rel_err < 1e-3, f"recovered {recovered:.6f} vs true {period:.6f} (rel_err={rel_err:.2e})"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_period_recovery_multiband(seed):
    period = 0.2734
    mjd, mag, err, fid = make_lightcurve(period, seed=seed, bands=("g", "r"))
    p = run_pipeline(mjd, mag, err, fid, largest_period=2.0)
    best_freq, _ = p.get_best_frequencies()
    recovered = 1.0 / best_freq[0]
    assert abs(recovered - period) / period < 1e-3


def test_api_contract(multi_band_lc):
    """The full public API used by the pipeline must keep working and return
    consistently-shaped results."""
    period, mjd, mag, err, fid = multi_band_lc
    p = run_pipeline(mjd, mag, err, fid, largest_period=2.0)

    best_freq, best_per = p.get_best_frequencies()
    assert best_freq.shape == best_per.shape
    assert best_freq.ndim == 1 and best_freq.size >= 1

    freq, per = p.get_periodogram()
    assert freq.shape == per.shape
    assert freq.size > 1000  # full grid

    # per-band best frequency
    for band in ("g", "r"):
        fb = p.get_best_frequency(band)
        assert np.isfinite(fb) and fb > 0

    # best frequency is the global argmax of the periodogram
    assert np.isclose(p.get_best_frequency(), freq[np.argmax(per)])


def test_finetune_improves_resolution(single_band_lc):
    """Fine-tuning should land on (or improve) the coarse-grid peak, never make
    the best periodogram value worse."""
    period, mjd, mag, err, fid = single_band_lc
    p = MultiBandPeriodogram(method="MHAOV")
    p.set_data(mjd, mag, err, fid)
    p.optimal_frequency_grid_evaluation(smallest_period=0.02, largest_period=2.0, shift=0.1)
    coarse_best = p.per.max()
    p.optimal_finetune_best_frequencies(times_finer=10.0, n_local_optima=10)
    best_freq, best_per = p.get_best_frequencies()
    assert best_per[0] >= coarse_best - 1e-3
    assert abs(1.0 / best_freq[0] - period) / period < 1e-3


def test_astropy_crosscheck():
    """Dominant frequency must agree with an independent Lomb-Scargle estimate."""
    astropy = pytest.importorskip("astropy")
    from astropy.timeseries import LombScargle

    period = 0.2734
    mjd, mag, err, fid = make_lightcurve(period, seed=3, bands=("g",))
    p = run_pipeline(mjd, mag, err, fid, largest_period=2.0)
    best_freq, _ = p.get_best_frequencies()
    p4j_period = 1.0 / best_freq[0]

    freq_grid = np.linspace(1.0 / 2.0, 1.0 / 0.02, 200000)
    power = LombScargle(mjd, mag, err).power(freq_grid)
    ls_period = 1.0 / freq_grid[np.argmax(power)]

    assert abs(p4j_period - ls_period) / ls_period < 1e-2
