from mongo_scribe.sql.command.parser import multistream_detection_to_ztf

def test_conversion_to_ztf():
    incoming_command = {
        "aid": "aid",
        "oid": "ZTF123",
        "sid": "ZTF",
        "pid": 123123,
        "tid": "ZTF",
        "fid": "g",
        "candid": "8276382778",
        "mjd": 55000,
        "ra": 45.0,
        "dec": 45.0,
        "mag": 15.11,
        "e_mag": 1.0,
        "mag_corr": None,
        "e_mag_corr": None,
        "e_mag_corr_ext": None,
        "isdiffpos": 0,
        "corrected": False,
        "stellar": False,
        "dubious": False,
        "has_stamp": False,
        "parent_candid": None,
        "extra_fields": {}
    }

    expected_output = {
        "oid": "ZTF123",
        "fid": 1,
        "candid": 8276382778,
        "mjd": 55000,
        "ra": 45.0,
        "dec": 45.0,
        "magpsf": 15.11,
        "sigmapsf": 1.0,
        "magpsf_corr": None,
        "sigmapsf_corr": None,
        "sigmapsf_corr_ext": None,
        "isdiffpos": 0,
        "corrected": False,
        "stellar": False,
        "dubious": False,
        "has_stamp": False,
        "parent_candid": None
    }

    output = multistream_detection_to_ztf(incoming_command)
    assert output == expected_output