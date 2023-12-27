"""
This test file aims to check if the step keeps detections from the stream rather than from
the database. If a detection is in the stream and also in the database there are 5 options.

1. The detection is an alert (has stamp) and is being inserted for the first time
        It should then be kept as it is
2. The detection is an alert and it is being reprocessed
        The one from the database should be discarded
3. The detection is a prv detection (does not have stamp) and is being inserted for the first time
        It should then be kept as it is
4. The detection is an alert but came as a prv detection (does not have stamp)
        The one from the database should be kept
5. The detection is a prv detection and it is being reprocessed
        The one from the database should be discarded
"""

from unittest import mock
from .utils.mock_secret import mock_get_secret


@mock.patch("credentials.get_secret")
def test_alert_first_time(
    mock_credentials,
    produce_messages,
    mongo_service,
    env_variables,
    psql_conn,
    mongo_conn,
    kafka_consumer,
    insert_object,
    insert_non_detection,
    insert_forced_photometry,
    insert_detection,
):
    from scripts.run_step import step_creator

    insert_object(
        {"oid": "ZTF000llmn", "aid": "AL00XYZ00"},
        sql=psql_conn,
        mongo=mongo_conn,
    )
    insert_detection(
        {
            "oid": "ZTF000llmn",
            "aid": "AL00XYZ00",
            "candid": "123",
            "has_stamp": True,
        },
        sql=psql_conn,
        mongo=mongo_conn,
    )

    mock_credentials.side_effect = mock_get_secret
    detections = [
        {
            "aid": "AL00XYZ00",
            "oid": "ZTF000llmn",
            "sid": "ztf",
            "tid": "ztf",
            "pid": 555,
            "fid": "g",
            "candid": "456",
            "mjd": 55000,
            "ra": 45,
            "e_ra": 0.1,
            "dec": 45,
            "e_dec": 0.1,
            "mag": 23.1,
            "e_mag": 0.9,
            "isdiffpos": 1,
            "has_stamp": True,
            "forced": False,
            "extra_fields": {},
        },
        {
            "aid": "AL00XYZ00",
            "oid": "ZTF000llmn",
            "sid": "ztf",
            "tid": "ztf",
            "pid": 555,
            "fid": "g",
            "candid": "ZTF000llmn555",
            "mjd": 55000,
            "ra": 45,
            "e_ra": 0.1,
            "dec": 45,
            "e_dec": 0.1,
            "mag": 23.1,
            "e_mag": 0.9,
            "isdiffpos": 1,
            "has_stamp": False,
            "forced": True,
            "extra_fields": {},
        },
    ]
    produce_messages(
        "correction", 1, detections=detections, oids=["ZTF000llmn"]
    )
    step_creator().start()

    consumer = kafka_consumer(["lightcurve"])
    messages = list(consumer.consume())
    assert len(messages) == 1
    msg = messages[0]
    detections = msg["detections"]
    assert len(detections) == 3
    assert all([det["new"] for det in detections if det["candid"] != "123"])
    assert not all(
        [det["new"] for det in detections if det["candid"] == "123"]
    )


@mock.patch("credentials.get_secret")
def test_alert_reprocess(
    mock_credentials,
    produce_messages,
    mongo_service,
    env_variables,
    psql_conn,
    mongo_conn,
    kafka_consumer,
    insert_object,
    insert_non_detection,
    insert_forced_photometry,
    insert_detection,
):
    from scripts.run_step import step_creator

    insert_object(
        {"oid": "ZTF000llmn", "aid": "AL00XYZ00"},
        sql=psql_conn,
        mongo=mongo_conn,
    )
    insert_detection(
        {
            "oid": "ZTF000llmn",
            "aid": "AL00XYZ00",
            "candid": "123",
            "has_stamp": True,
        },
        sql=psql_conn,
        mongo=mongo_conn,
    )

    mock_credentials.side_effect = mock_get_secret
    detections = [
        {
            "aid": "AL00XYZ00",
            "oid": "ZTF000llmn",
            "sid": "ztf",
            "tid": "ztf",
            "pid": 555,
            "fid": "g",
            "candid": "123",
            "mjd": 55000,
            "ra": 45,
            "e_ra": 0.1,
            "dec": 45,
            "e_dec": 0.1,
            "mag": 23.1,
            "e_mag": 0.9,
            "isdiffpos": 1,
            "has_stamp": True,
            "forced": False,
            "extra_fields": {},
        },
    ]
    produce_messages(
        "correction", 1, detections=detections, oids=["ZTF000llmn"]
    )
    step_creator().start()

    consumer = kafka_consumer(["lightcurve"])
    messages = list(consumer.consume())
    assert len(messages) == 1
    msg = messages[0]
    detections = msg["detections"]
    assert len(detections) == 1
    det = detections[0]
    assert det["candid"] == "123"
    assert det["has_stamp"]
    assert det["new"]


@mock.patch("credentials.get_secret")
def test_alert_prv_detection(
    mock_credentials,
    produce_messages,
    mongo_service,
    env_variables,
    psql_conn,
    mongo_conn,
    kafka_consumer,
    insert_object,
    insert_non_detection,
    insert_forced_photometry,
    insert_detection,
):
    from scripts.run_step import step_creator

    mock_credentials.side_effect = mock_get_secret
    detections = [
        {
            "aid": "AL00XYZ00",
            "oid": "ZTF000llmn",
            "sid": "ztf",
            "tid": "ztf",
            "pid": 555,
            "fid": "g",
            "candid": "123",
            "mjd": 55000,
            "ra": 45,
            "e_ra": 0.1,
            "dec": 45,
            "e_dec": 0.1,
            "mag": 23.1,
            "e_mag": 0.9,
            "isdiffpos": 1,
            "has_stamp": True,
            "forced": False,
            "extra_fields": {},
        },
        {
            "aid": "AL00XYZ00",
            "oid": "ZTF000llmn",
            "sid": "ztf",
            "tid": "ztf",
            "pid": 555,
            "fid": "g",
            "candid": "456",
            "mjd": 55000,
            "ra": 45,
            "e_ra": 0.1,
            "dec": 45,
            "e_dec": 0.1,
            "mag": 23.1,
            "e_mag": 0.9,
            "isdiffpos": 1,
            "has_stamp": False,
            "forced": False,
            "extra_fields": {},
        },
    ]
    produce_messages(
        "correction", 1, detections=detections, oids=["ZTF000llmn"]
    )
    step_creator().start()

    consumer = kafka_consumer(["lightcurve"])
    messages = list(consumer.consume())
    assert len(messages) == 1
    msg = messages[0]
    detections = msg["detections"]
    assert len(detections) == 2
    det = detections[0]
    assert det["candid"] == "123"
    assert det["has_stamp"]
    det = detections[1]
    assert det["candid"] == "456"
    assert not det["has_stamp"]


@mock.patch("credentials.get_secret")
def test_alert_prv_detection_alert(
    mock_credentials,
    produce_messages,
    mongo_service,
    env_variables,
    psql_conn,
    mongo_conn,
    kafka_consumer,
    insert_object,
    insert_non_detection,
    insert_forced_photometry,
    insert_detection,
):
    from scripts.run_step import step_creator

    insert_object(
        {"oid": "ZTF000llmn", "aid": "AL00XYZ00"},
        sql=psql_conn,
        mongo=mongo_conn,
    )
    insert_detection(
        {
            "oid": "ZTF000llmn",
            "aid": "AL00XYZ00",
            "candid": "123",
            "has_stamp": True,
        },
        sql=psql_conn,
        mongo=mongo_conn,
    )

    mock_credentials.side_effect = mock_get_secret
    detections = [
        {
            "aid": "AL00XYZ00",
            "oid": "ZTF000llmn",
            "sid": "ztf",
            "tid": "ztf",
            "pid": 555,
            "fid": "g",
            "candid": "123",
            "mjd": 55000,
            "ra": 45,
            "e_ra": 0.1,
            "dec": 45,
            "e_dec": 0.1,
            "mag": 23.1,
            "e_mag": 0.9,
            "isdiffpos": 1,
            "has_stamp": False,
            "forced": False,
            "extra_fields": {},
        },
        {
            "aid": "AL00XYZ00",
            "oid": "ZTF000llmn",
            "sid": "ztf",
            "tid": "ztf",
            "pid": 555,
            "fid": "g",
            "candid": "ZTF000llmn555",
            "mjd": 55000,
            "ra": 45,
            "e_ra": 0.1,
            "dec": 45,
            "e_dec": 0.1,
            "mag": 23.1,
            "e_mag": 0.9,
            "isdiffpos": 1,
            "has_stamp": False,
            "forced": True,
            "extra_fields": {},
        },
    ]
    produce_messages(
        "correction", 1, detections=detections, oids=["ZTF000llmn"]
    )
    step_creator().start()

    consumer = kafka_consumer(["lightcurve"])
    messages = list(consumer.consume())
    assert len(messages) == 1
    msg = messages[0]
    detections = msg["detections"]
    assert len(detections) == 2
    det = detections[0]
    assert det["candid"] == "123"
    assert det["has_stamp"]
    det = detections[1]
    assert det["candid"] == "ZTF000llmn555"
    assert not det["has_stamp"]
    assert det["forced"]


@mock.patch("credentials.get_secret")
def test_alert_prv_detection_alert_reprocess(
    mock_credentials,
    produce_messages,
    mongo_service,
    env_variables,
    psql_conn,
    mongo_conn,
    kafka_consumer,
    insert_object,
    insert_non_detection,
    insert_forced_photometry,
    insert_detection,
):
    from scripts.run_step import step_creator

    insert_object(
        {"oid": "ZTF000llmn", "aid": "AL00XYZ00"},
        sql=psql_conn,
        mongo=mongo_conn,
    )
    insert_detection(
        {
            "oid": "ZTF000llmn",
            "aid": "AL00XYZ00",
            "candid": "123",
            "has_stamp": True,
        },
        sql=psql_conn,
        mongo=mongo_conn,
    )
    insert_detection(
        {
            "oid": "ZTF000llmn",
            "aid": "AL00XYZ00",
            "candid": "456",
            "has_stamp": False,
        },
        sql=psql_conn,
        mongo=mongo_conn,
    )

    mock_credentials.side_effect = mock_get_secret
    detections = [
        {
            "aid": "AL00XYZ00",
            "oid": "ZTF000llmn",
            "sid": "ztf",
            "tid": "ztf",
            "pid": 555,
            "fid": "g",
            "candid": "123",
            "mjd": 55000,
            "ra": 45,
            "e_ra": 0.1,
            "dec": 45,
            "e_dec": 0.1,
            "mag": 23.1,
            "e_mag": 0.9,
            "isdiffpos": 1,
            "has_stamp": True,
            "forced": False,
            "extra_fields": {},
        },
        {
            "aid": "AL00XYZ00",
            "oid": "ZTF000llmn",
            "sid": "ztf",
            "tid": "ztf",
            "pid": 555,
            "fid": "g",
            "candid": "456",
            "mjd": 55000,
            "ra": 45,
            "e_ra": 0.1,
            "dec": 45,
            "e_dec": 0.1,
            "mag": 23.1,
            "e_mag": 0.9,
            "isdiffpos": 1,
            "has_stamp": False,
            "forced": False,
            "extra_fields": {},
        },
    ]
    produce_messages(
        "correction", 1, detections=detections, oids=["ZTF000llmn"]
    )
    step_creator().start()

    consumer = kafka_consumer(["lightcurve"])
    messages = list(consumer.consume())
    assert len(messages) == 1
    msg = messages[0]
    detections = msg["detections"]
    assert len(detections) == 2
    det = detections[0]
    assert det["candid"] == "123"
    assert det["has_stamp"]
    det = detections[1]
    assert det["candid"] == "456"
    assert not det["has_stamp"]
    assert not det["forced"]
