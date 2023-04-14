import pandas as pd
from unittest import mock
from mocks.data import mock_lightcurve, mock_lightcurves_list, mock_xmatch_list

from xmatch_step.core.parsing import unparse, parse_output


def test_unparse():
    lc = pd.DataFrame.from_records(mock_lightcurve, index="aid")
    unparsed = unparse(lc, "detections")
    unparsed = unparsed["detections"].to_dict()
    detections = unparsed["aid"]
    assert len(detections) == 1  # ATLAS detection was ignored
    keys = detections[0].keys()
    assert "extra_fields" not in keys
    assert all([elem in keys for elem in ["magpsf", "sigmapsf"]])


def test_parse_output():
    lightcurve_dataframe = pd.DataFrame.from_records(mock_lightcurves_list)
    xmatch_dataframe = pd.DataFrame.from_records(mock_xmatch_list)
    result_dataframe = parse_output(lightcurve_dataframe, xmatch_dataframe)

    assert len(result_dataframe) == 2
    assert "xmatches" in result_dataframe[0].keys()
    assert "xmatches" in result_dataframe[1].keys()
    assert "allwise" in result_dataframe[0]["xmatches"].keys()
    assert "allwise" in result_dataframe[0]["xmatches"].keys()
