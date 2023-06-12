import pandas as pd
from mocks.data import mock_lightcurves_list, mock_xmatch_list

from xmatch_step.core.parsing import parse_output
from xmatch_step.core.utils.extract_info import (
    extract_detections_from_messages,
)


def test_parse_output():
    lc_hash = extract_detections_from_messages(mock_lightcurves_list)
    lightcurve_dataframe = pd.DataFrame.from_records(
        mock_lightcurves_list, exclude=["detections", "non_detections"]
    )
    xmatch_dataframe = pd.DataFrame.from_records(mock_xmatch_list)
    result_json = parse_output(lightcurve_dataframe, xmatch_dataframe, lc_hash)

    assert len(result_json) == 2
    assert result_json[0]["detections"] is not None
    assert result_json[0]["non_detections"] == []
    assert result_json[1]["detections"] is not None
    assert result_json[1]["non_detections"] is not None
    assert "xmatches" in result_json[0].keys()
    assert "xmatches" in result_json[1].keys()
    assert "allwise" in result_json[0]["xmatches"].keys()
    assert "allwise" in result_json[0]["xmatches"].keys()
