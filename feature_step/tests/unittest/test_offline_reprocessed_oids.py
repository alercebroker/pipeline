"""A list of ZTF names becomes the oid array a batch run consumes.

The runner speaks multisurvey bigints, ascending; an external list (the
reprocessing campaign's parquet, a colleague's text file) speaks ZTF names,
unsorted, with repeats and nulls. The conversion has to agree with idmapper
byte for byte -- a wrong oid is not an error, it is a silent "no detections" --
and the array must come out unique and ascending, because unit index N means
oids[N*unit_size:...] of one specific array.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

PIPE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PIPE / "feature_step" / "scripts"))
sys.path.insert(0, str(PIPE / "libs" / "idmapper"))

import offline_reprocessed_oids as M  # noqa: E402
from idmapper.mapper import catalog_oid_to_masterid  # noqa: E402
from idmapper.mapper import decode_masterid  # noqa: E402

SAMPLE = ["ZTF17aaaaaak", "ZTF17aaabauy", "ZTF18abjgybv", "ZTF26abtnngf",
          "ZTF00aaaaaaa", "ZTF99zzzzzzz"]


def test_encode_matches_idmapper_and_roundtrips():
    got = M.encode_ztf_ids(np.array(SAMPLE))
    want = np.array([catalog_oid_to_masterid("ZTF", s, validate=True) for s in SAMPLE],
                    dtype=np.int64)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, want)
    assert [decode_masterid(int(x))[1] for x in got] == SAMPLE


def test_known_value():
    # ZTF17aaabauy is the oid every offline CLI uses as its example.
    assert M.encode_ztf_ids(np.array(["ZTF17aaabauy"]))[0] == 36028933559755080


@pytest.mark.parametrize("bad", ["ZTF17aaabau", "ZTF17aaabauyy", "ztf17aaabauy",
                                 "ZTF1Xaaabauy", "ZTF17aaabaUy", "ZTF17aaab4uy", ""])
def test_invalid_names_are_rejected_by_name(bad):
    with pytest.raises(ValueError) as e:
        M.encode_ztf_ids(np.array(["ZTF17aaabauy", bad]))
    assert repr(bad) in str(e.value)


def test_txt_list_is_read_deduped_and_sorted(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("ZTF18abjgybv\n\nZTF17aaabauy\nZTF18abjgybv\n  \nZTF17aaaaaak\n")
    oids = M.build_oid_array(M.read_ztf_ids(p))
    want = M.encode_ztf_ids(np.array(["ZTF17aaaaaak", "ZTF17aaabauy", "ZTF18abjgybv"]))
    np.testing.assert_array_equal(oids, want)


def test_parquet_reads_one_column_and_drops_nulls(tmp_path):
    p = tmp_path / "alerts.parquet"
    t = pa.table({"oid": ["ZTF18abjgybv", None, "ZTF17aaabauy", "ZTF18abjgybv"],
                  "candid": [1, 2, 3, 4]})
    pq.write_table(t, p)
    ids = M.read_ztf_ids(p, column="oid")
    assert sorted(ids.tolist()) == ["ZTF17aaabauy", "ZTF18abjgybv", "ZTF18abjgybv"]
    oids = M.build_oid_array(ids)
    assert len(oids) == 2 and (np.diff(oids) > 0).all()


def test_eligible_filter_keeps_only_catalogue_oids_in_order():
    wanted = M.encode_ztf_ids(np.array(["ZTF17aaaaaak", "ZTF17aaabauy", "ZTF18abjgybv"]))
    catalogue = np.array([wanted[2], wanted[0], 12345], dtype=np.int64)  # unsorted, extra
    kept, n_dropped = M.keep_eligible(wanted, catalogue)
    np.testing.assert_array_equal(kept, np.sort(wanted[[0, 2]]))
    assert n_dropped == 1


def test_baseline_split_counts_new_and_repeated():
    oids = np.array([10, 20, 30, 40], dtype=np.int64)
    baseline = np.array([40, 20, 99], dtype=np.int64)
    n_new, n_repeat = M.split_against_baseline(oids, baseline)
    assert (n_new, n_repeat) == (2, 2)
