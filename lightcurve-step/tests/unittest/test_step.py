from unittest import mock
import pandas as pd
from pandas.testing import assert_frame_equal
import pickle
from lightcurve_step.step import LightcurveStep


def test_pre_execute_joins_detections_and_non_detections_and_adds_new_flag_to_detections():
    messages = [
        {
            "aid": "aid1",
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
                    "oid": "oid2",
                    "sid": "ztf",
                    "candid": "b",
                    "mjd": 2,
                    "has_stamp": True,
                    "extra_fields": {},
                },
            ],
            "non_detections": [{"mjd": 1, "oid": "i", "fid": "g"}],
        },
        {
            "aid": "aid1",
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
                {
                    "aid": "aid1",
                    "oid": "oid1",
                    "sid": "ztf",
                    "candid": "b",
                    "mjd": 2,
                    "has_stamp": True,
                    "extra_fields": {},
                },
            ],
            "non_detections": [{"mjd": 1, "oid": "i", "fid": "g"}],
        },
        {
            "aid": "aid2",
            "detections": [
                {
                    "aid": "aid2",
                    "oid": "oid6",
                    "sid": "ztf",
                    "candid": "c",
                    "mjd": 5,
                    "has_stamp": True,
                    "extra_fields": {},
                },
                {
                    "aid": "aid2",
                    "oid": "oid5",
                    "sid": "atlas",
                    "candid": "b",
                    "mjd": 4,
                    "has_stamp": True,
                    "extra_fields": {},
                },
            ],
            "non_detections": [{"mjd": 1, "oid": "i", "fid": "g"}],
        },
    ]

    output = LightcurveStep.pre_execute(messages)
    expected = {
        "aids": {"aid1", "aid2"},
        "oids": {
            "oid1": "aid1",
            "oid2": "aid1",
            "oid3": "aid1",
            "oid6": "aid2",
        },
        "last_mjds": {"aid1": 4, "aid2": 5},
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
                "oid": "oid2",
                "sid": "ztf",
                "candid": "b",
                "mjd": 2,
                "has_stamp": True,
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
                "aid": "aid1",
                "oid": "oid1",
                "sid": "ztf",
                "candid": "b",
                "mjd": 2,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
            {
                "aid": "aid2",
                "oid": "oid6",
                "sid": "ztf",
                "candid": "c",
                "mjd": 5,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
            {
                "aid": "aid2",
                "oid": "oid5",
                "sid": "atlas",
                "candid": "b",
                "mjd": 4,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
        ],
        "non_detections": [
            {"mjd": 1, "oid": "i", "fid": "g"},
            {"mjd": 1, "oid": "i", "fid": "g"},
            {"mjd": 1, "oid": "i", "fid": "g"},
        ],
    }
    assert output == expected


def test_execute_removes_duplicates_keeping_ones_with_stamps():
    mock_mongo= mock.MagicMock()
    mock_sql = mock.MagicMock()
    mock_get_sql_detections = mock.MagicMock()
    mock_get_sql_non_detections = mock.MagicMock()
    LightcurveStep.__init__ = lambda self: None
    step = LightcurveStep()
    step.db_mongo = mock_mongo
    step.db_sql = mock_sql
    step._get_sql_detections = mock_get_sql_detections
    step._get_sql_non_detections = mock_get_sql_non_detections
    step.logger = mock.MagicMock()
    mock_mongo.database["detection"].aggregate.return_value = [
        {
            "aid": "aid2",
            "candid": "d",
            "parent_candid": "p_d",
            "has_stamp": True,
            "sid": "SURVEY",
            "mjd": 1.0,
            "fid": "g",
            "new": False,
            "extra_fields": {},
        },
        {
            "aid": "aid1",
            "candid": 97923792234,
            "parent_candid": "p_a",
            "has_stamp": True,
            "sid": "SURVEY",
            "mjd": 1.0,
            "fid": "g",
            "new": False,
            "extra_fields": {},
        },
    ]
    mock_mongo.query.return_value.collection.find.return_value = []

    mock_get_sql_detections.return_value = [
        {
            "aid": "aid2",
            "oid": "oid2",
            "candid": "d",
            "parent_candid": "p_d",
            "has_stamp": True,
            "sid": "SURVEY",
            "mjd": 1.0,
            "fid": "g",
            "new": False,
            "extra_fields": {},
        },
        {
            "aid": "aid1",
            "oid": "oid1",
            "candid": "f",
            "parent_candid": "p_d",
            "has_stamp": True,
            "sid": "SURVEY",
            "mjd": 1.0,
            "fid": "g",
            "new": False,
            "extra_fields": {},
        },
    ]

    mock_get_sql_non_detections.return_value = [
        {
            "aid": "aid2",
            "fid": "g",
            "mjd": 2.0,
        },
        {
            "aid": "aid3",
            "fid": "f",
            "mjd": 3.0
        },
    ]    
    

    message = {
        "aids": {"aid1", "aid2"},
        "oids": {
            "oid1": "aid1",
            "oid2": "aid1",
            "oid3": "aid1",
            "oid2": "aid2",
        },
        "last_mjds": {"aid1": 4, "aid2": 5},
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
                "oid": "oid2",
                "sid": "ztf",
                "candid": "b",
                "mjd": 2,
                "has_stamp": True,
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
                "aid": "aid1",
                "oid": "oid1",
                "sid": "ztf",
                "candid": "b",
                "mjd": 2,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
            {
                "aid": "aid2",
                "oid": "oid2",
                "sid": "ztf",
                "candid": "c",
                "mjd": 5,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
            {
                "aid": "aid2",
                "oid": "oid2",
                "sid": "atlas",
                "candid": "b",
                "mjd": 4,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
        ],
        "non_detections": [
            {"mjd": 1, "oid": "i", "fid": "g"},
            {"mjd": 1, "oid": "i", "fid": "g"},
            {"mjd": 1, "oid": "i", "fid": "g"},
        ],
    }

    output = step.execute(message)
    print(output)
    expected = {
        "detections": [
            {
                "candid": "97923792234",
                "mjd": 1.0,
                "aid": "aid1",
                "parent_candid": "p_a",
                "has_stamp": True,
                "sid": "SURVEY",
                "fid": "g",
                "new": False,
                "extra_fields": {},
            },
            {
                "candid": "c",
                "mjd": 1.0,
                "aid": "aid2",
                "parent_candid": "p_c",
                "has_stamp": True,
                "sid": "SURVEY",
                "fid": "g",
                "new": True,
                "extra_fields": {},
            },
            {
                "candid": "b",
                "mjd": 1.0,
                "aid": "aid1",
                "parent_candid": "p_b",
                "has_stamp": False,
                "sid": "SURVEY",
                "fid": "g",
                "new": True,
                "extra_fields": {},
            },
            {
                "candid": "d",
                "mjd": 1.0,
                "aid": "aid2",
                "parent_candid": "p_d",
                "has_stamp": True,
                "sid": "SURVEY",
                "fid": "g",
                "new": False,
                "extra_fields": {},
            },
        ],
        "non_detections": [
            {"mjd": 1.0, "aid": "aid1", "fid": "g", "sid": "SURVEY", "tid": "SURVEY1"},
            {"mjd": 1.0, "aid": "aid2", "fid": "g", "sid": "SURVEY", "tid": "SURVEY1"},
        ],
        "last_mjds": {"aid1": 1.0, "aid2": 1.0},
    }

    exp_dets = pd.DataFrame(expected["detections"])
    exp_nd = pd.DataFrame(expected["non_detections"])

    assert_frame_equal(
        output["detections"].set_index("candid"),
        exp_dets.set_index("candid"),
        check_like=True,
    )
    assert_frame_equal(
        output["non_detections"].set_index(["aid", "fid", "mjd"]),
        exp_nd.set_index(["aid", "fid", "mjd"]),
        check_like=True,
    )


def test_pre_produce_restores_messages():
    message = {
        "detections": pd.DataFrame(
            [
                {
                    "candid": "a",
                    "mjd": 1,
                    "has_stamp": True,
                    "aid": "AID1",
                    "new": True,
                    "extra_fields": {"diaObject": b"bainari"},
                },
                {
                    "candid": "c",
                    "mjd": 1,
                    "new": True,
                    "has_stamp": True,
                    "aid": "AID2",
                    "extra_fields": {},
                },
                {
                    "candid": "MOST_RECENT",
                    "mjd": 2,
                    "has_stamp": True,
                    "aid": "AID1",
                    "new": False,
                    "extra_fields": {"diaObject": [{"a": "b"}]},
                },
                {
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
                {"mjd": 1, "oid": "a", "fid": 1, "aid": "AID1"},
                {"mjd": 1, "oid": "b", "fid": 1, "aid": "AID2"},
            ]
        ),
        "last_mjds": {"AID1": 1, "AID2": 1},
    }

    output = LightcurveStep.pre_produce(message)

    # This one has the object with MOST_RECENT candid removed
    expected = [
        {
            "aid": "AID1",
            "detections": [
                {
                    "candid": "a",
                    "has_stamp": True,
                    "mjd": 1,
                    "new": True,
                    "aid": "AID1",
                    "extra_fields": {"diaObject": b"bainari"},
                },
            ],
            "non_detections": [{"mjd": 1, "oid": "a", "fid": 1, "aid": "AID1"}],
        },
        {
            "aid": "AID2",
            "detections": [
                {
                    "candid": "c",
                    "mjd": 1,
                    "new": True,
                    "has_stamp": True,
                    "aid": "AID2",
                    "extra_fields": {},
                },
                {
                    "candid": "b",
                    "mjd": 1,
                    "new": False,
                    "has_stamp": False,
                    "aid": "AID2",
                    "extra_fields": {},
                },
            ],
            "non_detections": [{"mjd": 1, "oid": "b", "fid": 1, "aid": "AID2"}],
        },
    ]

    assert output == expected
