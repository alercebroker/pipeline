"""The worker's peak RSS has to land in the manifest to be worth anything.

Whether the 1.7 GB model stays shared copy-on-write under fork or degrades into
one copy per worker is the thing that ends a run, and today it is only visible
to somebody watching `top` while the probe runs.
"""
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PIPE / "feature_step" / "scripts"))

import offline_run_batch as R


def test_ru_maxrss_is_kilobytes_on_linux():
    assert R.rss_mb(2_048_000, "Linux") == 2000.0


def test_ru_maxrss_is_bytes_on_macos():
    """Same field, different unit — reading it raw makes a Mac look 1000x worse."""
    assert R.rss_mb(2_048_000 * 1024, "Darwin") == 2000.0


def test_the_manifest_carries_the_workers_peak_rss(monkeypatch, tmp_path):
    import pandas as pd

    cfg = {"out_dir": str(tmp_path), "minibatch": 500, "features": False,
           "credentials": "creds", "schema": "multisurvey_ztf", "retries": 0,
           "xmatch_url": "http://x", "min_detections": 1}
    inputs = ({oid: ({"oid": oid}, pd.DataFrame(),
                     pd.DataFrame([{"oid": oid, "W1": 1.0, "W2": 2.0,
                                    "W3": 3.0, "W4": 4.0}]))
               for oid in (1, 2)}, [])
    monkeypatch.setattr(R, "_W", {"cfg": cfg})
    monkeypatch.setattr(R, "fetch_minibatch", lambda mb, cfg: inputs)
    monkeypatch.setattr(R, "process_oid", lambda oid, *a, **k: ([{"oid": oid}], None))

    manifest = R.process_unit((0, [1, 2]))

    assert manifest["peak_rss_mb"] > 0
