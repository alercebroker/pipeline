import pandas as pd
import pytest
from mocks.data import mock_lightcurve

from xmatch_step.core.parsing import unparse


def test_unparse():
    lc = pd.DataFrame.from_records(mock_lightcurve, index="aid")
    unparsed = unparse(lc, "detections")
    unparsed = unparsed["detections"].to_dict()
    detections = unparsed["oid"]
    assert len(detections) == 1  # ATLAS detection was ignored
    keys = detections[0].keys()
    assert "extra_fields" not in keys
    assert all([elem in keys for elem in ["magpsf", "sigmapsf"]])


def test_unparse_empty():
    lc = mock_lightcurve.copy()
    lc[0]["detections"] = []
    empty_lc = pd.DataFrame.from_records(lc, index="aid")
    unparsed = unparse(empty_lc, "detections")
    assert unparsed.size == 0


def test_unparse_not_implemented():
    lc = pd.DataFrame.from_records(mock_lightcurve, index="aid")

    with pytest.raises(NotImplementedError) as ex:
        unparsed = unparse(lc, "oid")
    assert str(ex.value) == "Not implemented unparse for oid"


def parse_output():
    pass
