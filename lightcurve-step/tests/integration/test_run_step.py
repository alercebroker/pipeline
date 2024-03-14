from unittest import mock
from .utils.mock_secret import mock_get_secret


@mock.patch("credentials.get_secret")
def test_step_start(
    mock_credentials,
    produce_messages,
    mongo_service,
    env_variables,
    psql_conn,
    mongo_conn,
    kafka_consumer,
    insert_object,
    insert_detection,
):
    from scripts.run_step import step_creator

    mock_credentials.side_effect = mock_get_secret
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
        null_candid=True, # legacy schema
    )
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
    ]
    produce_messages(
        "correction", 1, detections=detections, oids=["ZTF000llmn"]
    )
    step_creator().start()
    consumer = kafka_consumer(["lightcurve"])
    messages = list(consumer.consume())
    assert len(messages) == 1
    for msg in messages:
        detections = msg["detections"]
        oids = set(map(lambda x: x["oid"], msg["detections"]))
        assert len(oids) == 1
        print(list(map(lambda x: x["candid"], msg["detections"])))
        assert len(detections) == 2
        candids_are_string = list(map(lambda x: type(x) == str, msg["candid"]))
        assert all(candids_are_string)
