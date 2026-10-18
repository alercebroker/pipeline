"""The reclassification runner's unit: counts, rows, write and manifest, with fakes.

fetch_minibatch and classify_minibatch are monkeypatched so no DB and no model
are needed; the taxonomy maps come from the fixture. What is under test is the
bookkeeping around them: which oids become rows, which are skipped and why,
that the write happens before the manifest, and that the stamped version is the
run's, not the module pin.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from alerce_classifiers.base.dto import OutputDTO
from features.offline.classifier_taxonomy_lut import TAXONOMY_LUT

PIPE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PIPE / "feature_step" / "scripts"))

import offline_reclassify_batch as B  # noqa: E402

TAX_MAPS = {cid: {name: idx for idx, name in enumerate(names)}
            for cid, names in TAXONOMY_LUT.items()}  # what fetch_taxonomy_maps returns


def _frame(classifier_id, oids):
    """A frame of the classifier's class labels, strictly decreasing probabilities, one row per oid."""
    names = TAXONOMY_LUT[classifier_id]
    probs = [1.0 / (i + 1) for i in range(len(names))]
    return pd.DataFrame([dict(zip(names, probs))] * len(oids), index=oids)


def _dto(oids):
    """A batched OutputDTO over `oids`, all five classifiers."""
    return OutputDTO(_frame(5, oids),
                     {"top": _frame(6, oids),
                      "children": {"Transient": _frame(7, oids),
                                   "Stochastic": _frame(8, oids),
                                   "Periodic": _frame(9, oids)}})


def _cfg(tmp_path, **over):
    cfg = {"out_dir": str(tmp_path), "minibatch": 500, "credentials": "creds",
           "schema": "multisurvey", "retries": 0, "warnings": False,
           "model_version": "2.2.0", "lastmjd_classifier": 5,
           "load_db": False, "write_credentials": None, "no_shards": False}
    cfg.update(over)
    return cfg


def _wire(monkeypatch, tmp_path, with_features, with_lastmjd, **over):
    """Fakes: oids in `with_features` get a frame, oids in `with_lastmjd` a lastmjd."""
    monkeypatch.setattr(B, "_W", {"cfg": _cfg(tmp_path, **over), "taxonomy": TAX_MAPS,
                                  "feature_list": ["Amplitude_1"]})
    monkeypatch.setattr(B, "fetch_minibatch",
                        lambda mb, cfg: (pd.DataFrame(), {o: 60000.5 for o in mb if o in with_lastmjd}))
    monkeypatch.setattr(B, "classify_minibatch",
                        lambda rows: (_dto(sorted(with_features)), sorted(with_features)))


def test_unit_classifies_oids_with_features_and_lastmjd(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, with_features={1, 2, 3}, with_lastmjd={1, 2, 3})
    manifest = B.process_unit((0, [1, 2, 3]))
    assert manifest["n_ok"] == 3
    assert manifest["prob_rows"] == 3 * 45          # 21 + 3 + 6 + 6 + 9 per oid
    assert manifest["n_skipped"] == 0
    shard = pd.read_parquet(tmp_path / "probabilities" / "unit_0000000.parquet")
    assert set(shard["classifier_version"]) == {220}   # the run's version, not the 210 pin
    assert set(shard["lastmjd"]) == {60000.5}


def test_unit_separates_no_features_from_no_lastmjd(monkeypatch, tmp_path):
    """Only an error is worth a retry, so the two silent skips get their own counters."""
    _wire(monkeypatch, tmp_path, with_features={1, 2}, with_lastmjd={1, 3})
    manifest = B.process_unit((0, [1, 2, 3]))
    assert manifest["n_ok"] == 1                    # oid 1: both
    assert manifest["n_no_lastmjd"] == 1            # oid 2: features, no stored row
    assert manifest["n_no_features"] == 1           # oid 3: no features at all
    assert manifest["n_skipped"] == 2
    assert not (tmp_path / "errors" / "unit_0000000.jsonl").exists()


def test_unit_writes_rows_before_the_manifest_and_counts_them(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, with_features={1}, with_lastmjd={1},
          load_db=True, write_credentials="wc", no_shards=True)
    seen = {}

    def _fake_write(rows, credentials, schema=None, execute=False):
        seen["rows"], seen["schema"], seen["execute"] = len(rows), schema, execute
        assert not (tmp_path / "manifests" / "unit_0000000.json").exists()  # write lands first
        return {"executed": True, "written": len(rows)}
    monkeypatch.setattr(B, "write_probabilities", _fake_write)

    manifest = B.process_unit((0, [1]))
    assert seen == {"rows": 45, "schema": "multisurvey", "execute": True}
    assert manifest["db_prob_rows"] == 45
    assert not (tmp_path / "probabilities").exists()  # --no-shards
    assert json.loads((tmp_path / "manifests" / "unit_0000000.json").read_text())["n_ok"] == 1


def test_unit_records_a_failed_oid_and_keeps_going(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, with_features={1, 2}, with_lastmjd={1, 2})
    real = B.build_probability_rows

    def _explode_on_two(dto, oid, *a, **k):
        if oid == 2:
            raise RuntimeError("boom 2")
        return real(dto, oid, *a, **k)
    monkeypatch.setattr(B, "build_probability_rows", _explode_on_two)

    manifest = B.process_unit((0, [1, 2]))
    assert manifest["n_ok"] == 1
    assert manifest["n_errors"] == 1
    recorded = json.loads((tmp_path / "errors" / "unit_0000000.jsonl").read_text())
    assert recorded["oid"] == 2 and "boom 2" in recorded["error"]


def test_unit_counts_an_incomplete_row_set_as_an_error(monkeypatch, tmp_path):
    """Every classified oid must yield exactly the taxonomy's row count (45); fewer is an error, not ok."""
    _wire(monkeypatch, tmp_path, with_features={1, 2}, with_lastmjd={1, 2})
    real = B.build_probability_rows

    def _short_on_two(dto, oid, *a, **k):
        rows = real(dto, oid, *a, **k)
        return rows[:36] if oid == 2 else rows        # oid 2 loses one branch's rows
    monkeypatch.setattr(B, "build_probability_rows", _short_on_two)

    manifest = B.process_unit((0, [1, 2]))
    assert manifest["n_ok"] == 1
    assert manifest["n_errors"] == 1
    assert manifest["prob_rows"] == 45                 # the short set was not written
    recorded = json.loads((tmp_path / "errors" / "unit_0000000.jsonl").read_text())
    assert recorded["oid"] == 2 and "36" in recorded["error"]
