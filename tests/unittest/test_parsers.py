import pandas as pd
from mocks.data import mock_lightcurves_list, mock_xmatch_list

from xmatch_step.core.parsing import parse_output


def test_parse_output():
    lightcurve_dataframe = pd.DataFrame.from_records(mock_lightcurves_list)
    xmatch_dataframe = pd.DataFrame.from_records(mock_xmatch_list)
    result_json = parse_output(lightcurve_dataframe, xmatch_dataframe)

    assert len(result_json) == 2
    assert result_json[0]["detections"] is not None
    assert result_json[0]["non_detections"] == []
    assert result_json[1]["detections"] is not None
    assert result_json[1]["non_detections"] is not None
    assert "xmatches" in result_json[0].keys()
    assert "xmatches" in result_json[1].keys()
    assert "allwise" in result_json[0]["xmatches"].keys()
    assert "allwise" in result_json[0]["xmatches"].keys()
