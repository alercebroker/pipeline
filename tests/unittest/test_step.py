from unittest import mock

from lightcurve_step.step import LightcurveStep


def test_pre_execute_joins_detections_and_non_detections():
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

    expected = {
        "aids": {"aid1", "aid2"},
        "detections": [
            {"candid": "b", "has_stamp": True},
            {"candid": "a", "has_stamp": False},
            {"candid": "c", "has_stamp": True},
            {"candid": "b", "has_stamp": False},
            {"candid": "c", "has_stamp": True},
            {"candid": "b", "has_stamp": False},
        ],
        "non_detections": [
            {"mjd": 1, "oid": "i", "fid": 1},
            {"mjd": 1, "oid": "i", "fid": 1},
            {"mjd": 1, "oid": "i", "fid": 1},
        ]
    }

    assert output == expected


def test_execute_removes_duplicates():
    mock_client = mock.MagicMock()
    LightcurveStep.__init__ = lambda self: None
    step = LightcurveStep()
    step.db_client = mock_client
    mock_client.query.return_value.collection.aggregate.return_value = []
    mock_client.query.return_value.collection.find.return_value = []

    message = {
        "aids": {"aid1", "aid2"},
        "detections": [
            {"candid": "a", "has_stamp": True},
            {"candid": "b", "has_stamp": False},
            {"candid": "c", "has_stamp": True},
            {"candid": "d", "has_stamp": False},
            {"candid": "d", "has_stamp": True},
            {"candid": "a", "has_stamp": False},
        ],
        "non_detections": [
            {"mjd": 1, "oid": "a", "fid": 1},
            {"mjd": 1, "oid": "b", "fid": 1},
            {"mjd": 1, "oid": "a", "fid": 1},
        ]
    }

    output = step.execute(message)

    expected = {
        "detections": [
            {"candid": "a", "has_stamp": True},
            {"candid": "c", "has_stamp": True},
            {"candid": "d", "has_stamp": True},
            {"candid": "b", "has_stamp": False},
        ],
        "non_detections": [
            {"mjd": 1, "oid": "a", "fid": 1},
            {"mjd": 1, "oid": "b", "fid": 1},
        ]
    }

    assert output == expected


def test_pre_produce_restores_messages():
    message = {
        "detections": [
            {"candid": "a", "has_stamp": True, "aid": "AID1"},
            {"candid": "c", "has_stamp": True, "aid": "AID2"},
            {"candid": "d", "has_stamp": True, "aid": "AID1"},
            {"candid": "b", "has_stamp": False, "aid": "AID2"},
        ],
        "non_detections": [
            {"mjd": 1, "oid": "a", "fid": 1, "aid": "AID1"},
            {"mjd": 1, "oid": "b", "fid": 1, "aid": "AID2"},
        ]
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
