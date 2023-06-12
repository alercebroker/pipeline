from unittest import mock

import pandas as pd
from pandas.testing import assert_frame_equal

from lightcurve_step.step import LightcurveStep


def test_pre_execute_joins_detections_and_non_detections_and_adds_new_flag_to_detections():
    messages = [
        {
            "aid": "aid1",
            "detections": [{"candid": "b", "has_stamp": True}, {"candid": "a", "has_stamp": False}],
            "non_detections": [{"mjd": 1, "oid": "i", "fid": "g"}]
        },
        {
            "aid": "aid1",
            "detections": [{"candid": "c", "has_stamp": True}, {"candid": "b", "has_stamp": False}],
            "non_detections": [{"mjd": 1, "oid": "i", "fid": "g"}]
        },
        {
            "aid": "aid2",
            "detections": [{"candid": "c", "has_stamp": True}, {"candid": "b", "has_stamp": False}],
            "non_detections": [{"mjd": 1, "oid": "i", "fid": "g"}]
        },
    ]

    output = LightcurveStep.pre_execute(messages)

    expected = {
        "aids": {"aid1", "aid2"},
        "detections": [
            {"candid": "b", "has_stamp": True, "new": True},
            {"candid": "a", "has_stamp": False, "new": True},
            {"candid": "c", "has_stamp": True, "new": True},
            {"candid": "b", "has_stamp": False, "new": True},
            {"candid": "c", "has_stamp": True, "new": True},
            {"candid": "b", "has_stamp": False, "new": True},
        ],
        "non_detections": [
            {"mjd": 1, "oid": "i", "fid": "g"},
            {"mjd": 1, "oid": "i", "fid": "g"},
            {"mjd": 1, "oid": "i", "fid": "g"},
        ]
    }

    assert output == expected


def test_execute_removes_duplicates_keeping_ones_with_stamps():
    mock_client = mock.MagicMock()
    LightcurveStep.__init__ = lambda self: None
    step = LightcurveStep()
    step.db_client = mock_client
    mock_client.query.return_value.collection.aggregate.return_value = [
        {"candid": "d", "has_stamp": True, "sid": "SURVEY", "fid": "g", "new": False},
        {"candid": "a", "has_stamp": True, "sid": "SURVEY", "fid": "g", "new": False},
    ]
    mock_client.query.return_value.collection.find.return_value = []

    message = {
        "aids": {"aid1", "aid2"},
        "detections": [
            {"candid": "a", "has_stamp": True, "sid": "SURVEY", "fid": "g", "new": True},
            {"candid": "b", "has_stamp": False, "sid": "SURVEY", "fid": "g", "new": True},
            {"candid": "c", "has_stamp": True, "sid": "SURVEY", "fid": "g", "new": True},
            {"candid": "d", "has_stamp": False, "sid": "SURVEY", "fid": "g", "new": True},
        ],
        "non_detections": [
            {"mjd": 1, "aid": "a", "fid": "g", "sid": "SURVEY", "tid": "SURVEY1"},
            {"mjd": 1, "aid": "b", "fid": "g", "sid": "SURVEY", "tid": "SURVEY1"},
            {"mjd": 1, "aid": "a", "fid": "g", "sid": "SURVEY", "tid": "SURVEY1"},
        ]
    }

    output = step.execute(message)

    expected = {
        "detections": [
            {"candid": "a", "has_stamp": True, "sid": "SURVEY", "fid": "g", "new": False},
            {"candid": "c", "has_stamp": True, "sid": "SURVEY", "fid": "g", "new": True},
            {"candid": "b", "has_stamp": False, "sid": "SURVEY", "fid": "g", "new": True},
            {"candid": "d", "has_stamp": True, "sid": "SURVEY", "fid": "g", "new": False},
        ],
        "non_detections": [
            {"mjd": 1, "aid": "a", "fid": "g", "sid": "SURVEY", "tid": "SURVEY1"},
            {"mjd": 1, "aid": "b", "fid": "g", "sid": "SURVEY", "tid": "SURVEY1"},
        ]
    }

    exp_dets = pd.DataFrame(expected["detections"])
    exp_nd = pd.DataFrame(expected["non_detections"])

    assert_frame_equal(output["detections"].set_index("candid"), exp_dets.set_index("candid"), check_like=True)
    assert_frame_equal(output["non_detections"].set_index(["aid", "fid", "mjd"]), exp_nd.set_index(["aid", "fid", "mjd"]), check_like=True)


def test_pre_produce_restores_messages():
    message = {
        "detections": pd.DataFrame([
            {"candid": "a", "has_stamp": True, "aid": "AID1"},
            {"candid": "c", "has_stamp": True, "aid": "AID2"},
            {"candid": "d", "has_stamp": True, "aid": "AID1"},
            {"candid": "b", "has_stamp": False, "aid": "AID2"},
        ]),
        "non_detections": pd.DataFrame([
            {"mjd": 1, "oid": "a", "fid": 1, "aid": "AID1"},
            {"mjd": 1, "oid": "b", "fid": 1, "aid": "AID2"},
        ])
    }

    output = LightcurveStep.pre_produce(message)

    expected = [
        {
            "aid": "AID1",
            "detections": [
                {"candid": "a", "has_stamp": True, "aid": "AID1"},
                {"candid": "d", "has_stamp": True, "aid": "AID1"},
            ],
            "non_detections": [{"mjd": 1, "oid": "a", "fid": 1, "aid": "AID1"}],
        },
        {
            "aid": "AID2",
            "detections": [
                {"candid": "c", "has_stamp": True, "aid": "AID2"},
                {"candid": "b", "has_stamp": False, "aid": "AID2"},
            ],
            "non_detections": [{"mjd": 1, "oid": "b", "fid": 1, "aid": "AID2"}],
        }
    ]

    assert output == expected
