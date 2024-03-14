from unittest import mock
import pandas as pd
from pandas.testing import assert_frame_equal
from lightcurve_step.step import LightcurveStep
from settings import settings_creator
import pickle


def test_pre_execute_joins_detections_and_non_detections_and_adds_new_flag_to_detections(
    env_variables,
):
    messages = [
        {
            "oid": "oid1",
            "candid": "a",
            "detections": [
                {
                    "aid": "aid1",
                    "oid": "oid1",
                    "sid": "ztf",
                    "candid": "a",
                    "mjd": 3,
                    "has_stamp": True,
                    "extra_fields": {},
                },
                {
                    "aid": "aid1",
                    "oid": "oid1",
                    "sid": "ztf",
                    "candid": "b",
                    "mjd": 2,
                    "has_stamp": False,
                    "extra_fields": {},
                },
            ],
            "non_detections": [{"mjd": 1, "oid": "oid1", "fid": "g"}],
        },
        {
            "oid": "oid3",
            "candid": "c",
            "detections": [
                {
                    "aid": "aid1",
                    "oid": "oid3",
                    "sid": "ztf",
                    "candid": "c",
                    "mjd": 4,
                    "has_stamp": True,
                    "extra_fields": {},
                },
            ],
            "non_detections": [{"mjd": 1, "oid": "oid3", "fid": "g"}],
        },
        {
            "oid": "oid2",
            "candid": "d",
            "detections": [
                {
                    "aid": "aid2",
                    "oid": "oid2",
                    "sid": "ztf",
                    "candid": "d",
                    "mjd": 5,
                    "has_stamp": True,
                    "extra_fields": {},
                },
                {
                    "aid": "aid2",
                    "oid": "oid2",
                    "sid": "ztf",
                    "candid": "e",
                    "mjd": 4,
                    "has_stamp": False,
                    "extra_fields": {},
                },
            ],
            "non_detections": [{"mjd": 1, "oid": "oid2", "fid": "g"}],
        },
    ]

    output = LightcurveStep(
        settings_creator(), mock.MagicMock(), mock.MagicMock()
    ).pre_execute(messages)
    expected = {
        "oids": ["oid1", "oid2", "oid3"],
        "candids": {"oid1": ["a"], "oid3": ["c"], "oid2": ["d"]},
        "last_mjds": {"oid1": 3, "oid2": 5, "oid3": 4},
        "detections": [
            {
                "aid": "aid1",
                "oid": "oid1",
                "sid": "ztf",
                "candid": "a",
                "mjd": 3,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
            {
                "aid": "aid1",
                "oid": "oid1",
                "sid": "ztf",
                "candid": "b",
                "mjd": 2,
                "has_stamp": False,
                "extra_fields": {},
                "new": True,
            },
            {
                "aid": "aid1",
                "oid": "oid3",
                "sid": "ztf",
                "candid": "c",
                "mjd": 4,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
            {
                "aid": "aid2",
                "oid": "oid2",
                "sid": "ztf",
                "candid": "d",
                "mjd": 5,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
            {
                "aid": "aid2",
                "oid": "oid2",
                "sid": "ztf",
                "candid": "e",
                "mjd": 4,
                "has_stamp": False,
                "extra_fields": {},
                "new": True,
            },
        ],
        "non_detections": [
            {"mjd": 1, "oid": "oid1", "fid": "g"},
            {"mjd": 1, "oid": "oid3", "fid": "g"},
            {"mjd": 1, "oid": "oid2", "fid": "g"},
        ],
    }
    expected_aids = expected.pop("oids")
    output_aids = output.pop("oids")
    assert set(expected_aids) == set(output_aids)
    assert output == expected


@mock.patch("lightcurve_step.step._get_sql_non_detections")
@mock.patch("lightcurve_step.step._get_sql_detections")
def test_execute_removes_duplicates_keeping_ones_with_stamps(
    _get_sql_det, _get_sql_non_det, env_variables
):
    mock_mongo = mock.MagicMock()
    db_mongo = mock_mongo
    db_sql = mock.MagicMock()
    step = LightcurveStep(settings_creator(), db_mongo, db_sql)
    mock_mongo.database["detection"].aggregate.return_value = [
        {
            "aid": "aid1",
            "candid": "d",
            "oid": "ZTF123",
            "parent_candid": "p_d",
            "has_stamp": True,
            "sid": "ZTF",
            "mjd": 1.0,
            "fid": "g",
            "new": False,
            "extra_fields": {},
        },
    ]
    mock_mongo.query.return_value.collection.find.return_value = []

    _get_sql_det.return_value = [
        {
            "aid": "aid1",
            "oid": "ZTF123",
            "candid": "d",
            "parent_candid": "p_d",
            "has_stamp": True,
            "sid": "ZTF",
            "mjd": 1.0,
            "fid": "g",
            "new": False,
            "extra_fields": {},
        },
        {
            "aid": "aid1",
            "oid": "ZTF123",
            "candid": "f",
            "parent_candid": "p_d",
            "has_stamp": True,
            "sid": "ZTF",
            "mjd": 1.0,
            "fid": "g",
            "new": False,
            "extra_fields": {},
        },
    ]

    _get_sql_non_det.return_value = [
        {
            "aid": "aid1",
            "sid": "ZTF",
            "tid": "ZTF",
            "oid": "ZTF123",
            "fid": "g",
            "mjd": 2.0,
            "diffmaglim": 1,
        },
        {
            "fid": "f",
            "aid": "aid1",
            "sid": "ZTF",
            "tid": "ZTF",
            "mjd": 3.0,
            "oid": "ZTF123",
            "diffmaglim": 1,
        },
    ]

    message = {
        "oids": ["ZTF123", "ZTF456"],
        "candids": {"ZTF123": ["d", "f"], "ZTF456": ["a"]},
        "last_mjds": {"ZTF123": 1, "ZTF456": 1},
        "detections": [
            {
                "aid": "aid1",
                "oid": "ZTF456",
                "sid": "ZTF",
                "parent_candid": "p_a",
                "candid": "a",
                "fid": "g",
                "mjd": 3,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
        ],
        "non_detections": [],
    }

    output = step.execute(message)
    expected = {
        "detections": [
            {
                "aid": "aid1",
                "oid": "ZTF456",
                "sid": "ZTF",
                "parent_candid": "p_a",
                "candid": "a",
                "fid": "g",
                "mjd": 3,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
            {
                "aid": "aid1",
                "oid": "ZTF123",
                "candid": "d",
                "parent_candid": "p_d",
                "has_stamp": True,
                "sid": "ZTF",
                "mjd": 1.0,
                "fid": "g",
                "new": False,
                "extra_fields": {},
            },
            {
                "aid": "aid1",
                "oid": "ZTF123",
                "candid": "f",
                "parent_candid": "p_d",
                "has_stamp": True,
                "sid": "ZTF",
                "mjd": 1.0,
                "fid": "g",
                "new": False,
                "extra_fields": {},
            },
        ],
        "non_detections": [
            {
                "aid": "aid1",
                "sid": "ZTF",
                "tid": "ZTF",
                "oid": "ZTF123",
                "fid": "g",
                "mjd": 2.0,
                "diffmaglim": 1,
            },
            {
                "fid": "f",
                "aid": "aid1",
                "sid": "ZTF",
                "tid": "ZTF",
                "mjd": 3.0,
                "oid": "ZTF123",
                "diffmaglim": 1,
            },
        ],
        "last_mjds": {"ZTF123": 1.0, "ZTF456": 1.0},
    }

    exp_dets = pd.DataFrame(expected["detections"])
    exp_nd = pd.DataFrame(expected["non_detections"])
    assert_frame_equal(
        output["detections"].set_index("candid"),
        exp_dets.set_index("candid"),
        check_like=True,
    )
    assert_frame_equal(
        output["non_detections"].set_index(["oid", "fid", "mjd"]),
        exp_nd.set_index(["oid", "fid", "mjd"]),
        check_like=True,
    )


def test_pre_produce_restores_messages(env_variables):
    message = {
        "detections": pd.DataFrame(
            [
                {
                    "oid": "oid1",
                    "candid": "a",
                    "mjd": 1,
                    "has_stamp": True,
                    "aid": "AID1",
                    "new": True,
                    "extra_fields": {"diaObject": b"bainari"},
                },
                {
                    "oid": "oid1",
                    "candid": "c",
                    "mjd": 1,
                    "new": True,
                    "has_stamp": True,
                    "aid": "AID2",
                    "extra_fields": {},
                },
                {
                    "oid": "oid1",
                    "candid": "MOST_RECENT",
                    "mjd": 2,
                    "has_stamp": True,
                    "aid": "AID1",
                    "new": True,
                    "extra_fields": {"diaObject": [{"a": "b"}]},
                },
                {
                    "oid": "oid2",
                    "candid": "b",
                    "mjd": 1,
                    "new": False,
                    "has_stamp": False,
                    "aid": "AID2",
                    "extra_fields": {},
                },
            ]
        ),
        "non_detections": pd.DataFrame(
            [
                {"mjd": 1, "oid": "oid1", "fid": 1, "aid": "AID1"},
                {"mjd": 1, "oid": "oid2", "fid": 1, "aid": "AID2"},
            ]
        ),
        "last_mjds": {"oid1": 2, "oid2": 1},
        "candids": {"oid1": ["a"], "oid2": ["b"]},
    }

    output = LightcurveStep(
        settings_creator(), mock.MagicMock(), mock.MagicMock()
    ).pre_produce(message)

    # This one has the object with MOST_RECENT candid removed
    expected = [
        {
            "oid": "oid1",
            "detections": [
                {
                    "oid": "oid1",
                    "candid": "a",
                    "mjd": 1,
                    "has_stamp": True,
                    "aid": "AID1",
                    "new": True,
                    "extra_fields": {"diaObject": b"bainari"},
                },
                {
                    "oid": "oid1",
                    "candid": "c",
                    "mjd": 1,
                    "new": True,
                    "has_stamp": True,
                    "aid": "AID2",
                    "extra_fields": {},
                },
                {
                    "oid": "oid1",
                    "candid": "MOST_RECENT",
                    "mjd": 2,
                    "has_stamp": True,
                    "aid": "AID1",
                    "new": True,
                    "extra_fields": {"diaObject": pickle.dumps([{"a": "b"}])},
                },
            ],
            "non_detections": [
                {"mjd": 1, "oid": "oid1", "fid": 1, "aid": "AID1"}
            ],
            "candid": ["a"],
        },
        {
            "oid": "oid2",
            "candid": ["b"],
            "detections": [
                {
                    "oid": "oid2",
                    "candid": "b",
                    "mjd": 1,
                    "new": False,
                    "has_stamp": False,
                    "aid": "AID2",
                    "extra_fields": {},
                },
            ],
            "non_detections": [
                {"mjd": 1, "oid": "oid2", "fid": 1, "aid": "AID2"}
            ],
        },
    ]

    assert output == expected
