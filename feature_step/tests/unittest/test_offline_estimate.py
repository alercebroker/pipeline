"""Projecting a full run from a probe's manifests.

The probe exists to replace arithmetic-on-a-laptop-measurement with numbers from
the target machine. These are the pure projections; reading the files is I/O.
"""
import sys
from pathlib import Path

import pytest

PIPE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PIPE / "feature_step" / "scripts"))

import offline_estimate as E


def _man(n_oids, elapsed_s, prob=0, feat=0, no_allwise=0, rss_mb=None):
    m = {"n_oids": n_oids, "n_ok": n_oids, "elapsed_s": elapsed_s,
         "prob_rows": prob, "feat_rows": feat, "n_no_allwise": no_allwise}
    if rss_mb is not None:
        m["peak_rss_mb"] = rss_mb
    return m


def test_hours_scale_with_total_oids_and_divide_by_workers():
    # 2 units x 100 oids in 200 s each = 400 core-s / 200 oids = 2 core-s/oid.
    # 1000 oids on 4 workers -> 2 * 1000 / 4 = 500 s.
    got = E.estimate([_man(100, 200.0), _man(100, 200.0)], n_total_oids=1000, workers=4)
    assert got["core_s_per_oid"] == pytest.approx(2.0)
    assert got["hours"] == pytest.approx(500 / 3600)


def test_row_totals_come_from_the_measured_rate_not_a_guess():
    got = E.estimate([_man(100, 1.0, prob=4500, feat=19300)],
                     n_total_oids=1_000_000, workers=1)
    assert got["prob_rows"] == 45_000_000      # 45/oid
    assert got["feat_rows"] == 193_000_000     # 193/oid


def test_the_slow_tail_is_reported_separately_from_the_mean():
    """A unit is 5000 oids in series, so the mean hides the long light curves
    that decide when the run's last worker finishes."""
    got = E.estimate([_man(100, 100.0)] * 9 + [_man(100, 1000.0)],
                     n_total_oids=1000, workers=1)
    assert got["unit_s_p50"] == pytest.approx(100.0)
    assert got["unit_s_max"] == pytest.approx(1000.0)


def test_projected_memory_is_the_worst_worker_times_the_worker_count():
    """What kills a run is N workers each holding their own copy of the model,
    so the projection has to be per-worker peak x workers, not a sum over units.
    """
    got = E.estimate([_man(100, 1.0, rss_mb=2000), _man(100, 1.0, rss_mb=2500)],
                     n_total_oids=1000, workers=64)
    assert got["peak_rss_mb"] == 2500
    assert got["projected_rss_gb"] == pytest.approx(2500 * 64 / 1024)


def test_memory_is_absent_rather_than_invented_when_the_probe_predates_it():
    got = E.estimate([_man(100, 1.0)], n_total_oids=1000, workers=64)
    assert got["peak_rss_mb"] is None and got["projected_rss_gb"] is None


def test_no_allwise_rate_is_carried_through():
    got = E.estimate([_man(100, 1.0, no_allwise=14)], n_total_oids=1000, workers=1)
    assert got["no_allwise_rate"] == pytest.approx(0.14)


def test_an_empty_probe_is_refused_rather_than_projected_from_nothing():
    with pytest.raises(ValueError, match="no manifests"):
        E.estimate([], n_total_oids=1000, workers=1)
