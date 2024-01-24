import pandas as pd
from mocks.data import mock_xmatch_list

from xmatch_step.core.parsing import parse_output


def test_parse_output():
    xmatch_dataframe = pd.DataFrame.from_records(mock_xmatch_list)
    candids = {"oid1": ["aaa"], "oid2": ["eee"]}
    lightcurve_by_oid = {
        "oid1": {
            "oid": "oid1",
            "detections": [
                {
                    "candid": "aaa",
                    "sid": "ZTF",
                    "tid": "ZTF",
                    "oid": "oid1",
                    "aid": "aid1",
                    "mag": 0.1,
                    "e_mag": 0.01,
                    "extra_fields": {"field1": "f", "field2": "f2"},
                },
            ],
            "non_detections": [],
        },
        "oid2": {
            "oid": "oid2",
            "detections": [
                {
                    "candid": "eee",
                    "sid": "ZTF",
                    "tid": "ZTF",
                    "oid": "oid2",
                    "aid": "aid2",
                    "mag": 0.1,
                    "e_mag": 0.01,
                    "extra_fields": {"field1": "f", "field2": "f2"},
                },
            ],
            "non_detections": [],
        },
    }
    result_json = parse_output(xmatch_dataframe, lightcurve_by_oid, candids)

    assert len(result_json) == 2
    assert result_json[0]["candid"] == ["aaa"]
    assert result_json[1]["candid"] == ["eee"]
    assert result_json[0]["detections"][0]["candid"] == "aaa"
    assert result_json[1]["detections"][0]["candid"] == "eee"
    assert result_json[0]["non_detections"] == []
    assert result_json[1]["non_detections"] == []
    assert "xmatches" in result_json[0].keys()
    assert "xmatches" in result_json[1].keys()
    assert "allwise" in result_json[0]["xmatches"].keys()
    assert "allwise" in result_json[1]["xmatches"].keys()
    assert isinstance(result_json[0]["xmatches"]["allwise"], dict)
    assert isinstance(result_json[1]["xmatches"]["allwise"], dict)
