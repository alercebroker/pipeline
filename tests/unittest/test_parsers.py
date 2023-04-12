import pandas as pd
from unittest import mock
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


def parse_output():
    pass
