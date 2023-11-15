from unittest import mock
import pandas as pd
from pandas.testing import assert_frame_equal
from lightcurve_step.step import LightcurveStep
from settings import settings_creator


def test_pre_execute_joins_detections_and_non_detections_and_adds_new_flag_to_detections(
    env_variables,
):
    messages = [
        {
            "aid": "aid1",
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
            "candid": "c",
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

    output = LightcurveStep(
        settings_creator(), mock.MagicMock(), mock.MagicMock()
    ).pre_execute(messages)
    expected = {
        "aids": ["aid1", "aid2"],
        "candids": {"aid1": ["a", "c"], "aid2": ["c"]},
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
    expected_aids = expected.pop("aids")
    output_aids = output.pop("aids")
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
            "aid": "aid2",
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
        {
            "aid": "aid1",
            "candid": 97923792234,
            "parent_candid": "p_a",
            "oid": "oidy",
            "has_stamp": True,
            "sid": "SURVEY",
            "mjd": 1.0,
            "fid": "g",
            "new": False,
            "extra_fields": {},
        },
    ]
    mock_mongo.query.return_value.collection.find.return_value = []

    _get_sql_det.return_value = [
        {
            "aid": "aid2",
            "oid": "ZTF123",
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
            "oid": "ZTF456",
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
            "aid": "aid2",
            "sid": "ZTF",
            "tid": "ZTF",
            "mjd": 3.0,
            "oid": "ZTF456",
            "diffmaglim": 1,
        },
    ]

    message = {
        "aids": {"aid1", "aid2"},
        "oids": {
            "ZTF456": "aid1",
            "ZTF123": "aid2",
            "oidy": "aid1",
        },
        "candids": {"aid1": ["a", "c"], "aid2": ["c"]},
        "last_mjds": {"aid1": 4, "aid2": 5},
        "detections": [
            {
                "aid": "aid1",
                "oid": "oid1",
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
                "aid": "aid2",
                "oid": "oid3",
                "sid": "ZTF",
                "candid": "c",
                "parent_candid": "p_c",
                "fid": "g",
                "mjd": 4,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
            {
                "aid": "aid1",
                "oid": "oid1",
                "sid": "ZTF",
                "parent_candid": "p_b",
                "fid": "g",
                "candid": "b",
                "mjd": 2,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
            {
                "aid": "aid2",
                "oid": "oid2",
                "sid": "ATLAS",
                "fid": "g",
                "candid": "b",
                "mjd": 4,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
        ],
        "non_detections": [
            {
                "mjd": 1,
                "oid": "oid1",
                "fid": "c",
                "aid": "aid1",
                "diffmaglim": 0.8,
                "sid": "ATLAS",
                "tid": "ATLAS-01a",
            },
        ],
    }

    output = step.execute(message)
    expected = {
        "detections": [
            {
                "oid": "oidy",
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
                "mjd": 4.0,
                "oid": "oid3",
                "aid": "aid2",
                "parent_candid": "p_c",
                "has_stamp": True,
                "sid": "ZTF",
                "fid": "g",
                "new": True,
                "extra_fields": {},
            },
            {
                "candid": "b",
                "mjd": 2.0,
                "oid": "oid1",
                "aid": "aid1",
                "parent_candid": "p_b",
                "has_stamp": True,
                "sid": "ZTF",
                "fid": "g",
                "new": True,
                "extra_fields": {},
            },
            {
                "candid": "d",
                "mjd": 1.0,
                "oid": "ZTF123",
                "aid": "aid2",
                "parent_candid": "p_d",
                "has_stamp": True,
                "sid": "ZTF",
                "fid": "g",
                "new": False,
                "extra_fields": {},
            },
            {
                "aid": "aid1",
                "oid": "ZTF456",
                "candid": "f",
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
                "oid": "oid1",
                "fid": "g",
                "sid": "ZTF",
                "parent_candid": "p_a",
                "candid": "a",
                "mjd": 3,
                "has_stamp": True,
                "extra_fields": {},
                "new": True,
            },
        ],
        "non_detections": [
            {
                "mjd": 1.0,
                "aid": "aid1",
                "oid": "oid1",
                "diffmaglim": 0.8,
                "fid": "c",
                "sid": "ATLAS",
                "tid": "ATLAS-01a",
            },
            {
                "mjd": 2.0,
                "aid": "aid1",
                "oid": "ZTF123",
                "diffmaglim": 1,
                "fid": "g",
                "sid": "ZTF",
                "tid": "ZTF",
            },
            {
                "mjd": 3.0,
                "aid": "aid2",
                "oid": "ZTF456",
                "diffmaglim": 1,
                "fid": "f",
                "sid": "ZTF",
                "tid": "ZTF",
            },
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


def test_pre_produce_restores_messages(env_variables):
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
        "candids": {"AID1": ["a", "MOST_RECENT"], "AID2": ["c", "b"]},
    }

    output = LightcurveStep(
        settings_creator(), mock.MagicMock(), mock.MagicMock()
    ).pre_produce(message)

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
            "candid": ["a", "MOST_RECENT"],
        },
        {
            "aid": "AID2",
            "candid": ["c", "b"],
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
