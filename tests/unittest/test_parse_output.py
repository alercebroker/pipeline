from pandas import Series
from sorting_hat_step.utils.output import parse_output


def test_parse_output_ztf():
    data = {
        "oid": "ZTF_ALERT1",
        "tid": "ZTF",
        "pid": "3",
        "candid": 123,
        "mjd": 123,
        "fid": 123,
        "ra": 123,
        "dec": 123,
        "rb": 123,
        "rbversion": "v1",
        "mag": 123,
        "e_mag": 123,
        "rfid": 123,
        "isdiffpos": 0,
        "e_ra": 123,
        "e_dec": 123,
        "extra_fields": {"prv_candidates": None},
        "stamps": {
            "cutoutScience": b"science",
            "cutoutDifference": b"difference",
            "cutoutTemplate": b"template",
        },
        "aid": "aid",
    }
    inp = Series(data=data, index=data.keys())
    result = parse_output(inp)
    assert result["stamps"]["science"] == b"science"
    assert result["stamps"]["difference"] == b"difference"
    assert result["stamps"]["template"] == b"template"


def test_parse_output_atlas():
    data = {
        "oid": "ATLAS_ALERT1",
        "tid": "ATLAS123",
        "pid": "3",
        "candid": 123,
        "mjd": 123,
        "fid": 123,
        "ra": 123,
        "dec": 123,
        "rb": 123,
        "rbversion": "v1",
        "mag": 123,
        "e_mag": 123,
        "rfid": 123,
        "isdiffpos": 0,
        "e_ra": 123,
        "e_dec": 123,
        "extra_fields": {"prv_candidates": None},
        "stamps": {
            "cutoutScience": b"science",
            "cutoutDifference": b"difference",
            "cutoutTemplate": b"template",
        },
        "aid": "aid",
    }
    inp = Series(data=data, index=data.keys())
    result = parse_output(inp)
    assert result["stamps"]["science"] == b"science"
    assert result["stamps"]["difference"] == b"difference"
    assert result["stamps"]["template"] == None
