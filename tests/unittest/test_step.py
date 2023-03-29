from lightcurve_step.step import LightcurveStep


def test_unique_detections_joins_non_repeated_detections_based_on_candid_keeping_with_stamp():
    old = [{"candid": "a", "has_stamp": False}, {"candid": "b", "has_stamp": True}]
    new = [{"candid": "a", "has_stamp": True}, {"candid": "c", "has_stamp": False}]

    result = LightcurveStep.unique_detections(old, new)
    assert result == [{"candid": "b", "has_stamp": True}, {"candid": "a", "has_stamp": True}, {"candid": "c", "has_stamp": False}]


def test_unique_non_detections_joins_non_repeated_detections_based_on_mjd_oid_and_fid_keeping_first_member():
    old = [{"mjd": 1, "oid": "a", "fid": 1}, {"mjd": 1, "oid": "a", "fid": 2}]
    new = [{"mjd": 1, "oid": "a", "fid": 1, "source": "new"}, {"mjd": 1, "oid": "b", "fid": 2}]

    result = LightcurveStep.unique_non_detections(old, new)
    assert result == [{"mjd": 1, "oid": "a", "fid": 1}, {"mjd": 1, "oid": "a", "fid": 2}, {"mjd": 1, "oid": "b", "fid": 2}]


def test_clean_detections_from_db_adds_removes_id_field_and_adds_candid_based_on_its_value_if_not_present():
    from_db = [{"_id": "a"}, {"_id": "c", "candid": "b"}]
    LightcurveStep.clean_detections_from_db(from_db)
    assert from_db == [{"candid": "a"}, {"candid": "b"}]


def test_clean_non_detections_from_db_removes_id_field():
    from_db = [{"_id": "a"}, {"_id": "c", "candid": "b"}]
    LightcurveStep.clean_non_detections_from_db(from_db)
    assert from_db == [{}, {"candid": "b"}]


def test_pre_execute_joins_same_aids_into_single_message():
    messages = [
        {
            "aid": "aid1",
            "detections": [{"candid": "b", "has_stamp": True}, {"candid": "a", "has_stamp": False}],
            "non_detections": [{"mjd": 1, "oid": "i", "fid": 1}]
        },
        {
            "aid": "aid1",
            "detections": [{"candid": "c", "has_stamp": True}, {"candid": "b", "has_stamp": False}],
            "non_detections": [{"mjd": 1, "oid": "i", "fid": 1}]
        },
        {
            "aid": "aid2",
            "detections": [{"candid": "c", "has_stamp": True}, {"candid": "b", "has_stamp": False}],
            "non_detections": [{"mjd": 1, "oid": "i", "fid": 1}]
        },
    ]

    output = LightcurveStep.pre_execute(messages)

    expected = [
        {
            "aid": "aid1",
            "detections": [{"candid": "b", "has_stamp": True}, {"candid": "a", "has_stamp": False}, {"candid": "c", "has_stamp": True}],
            "non_detections": [{"mjd": 1, "oid": "i", "fid": 1}]
        },
        {
            "aid": "aid2",
            "detections": [{"candid": "c", "has_stamp": True}, {"candid": "b", "has_stamp": False}],
            "non_detections": [{"mjd": 1, "oid": "i", "fid": 1}]
        },
    ]

    assert output == expected
