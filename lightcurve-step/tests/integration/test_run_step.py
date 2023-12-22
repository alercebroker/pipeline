from unittest import mock
import json
import os


def mock_get_secret(secret_name):
    if secret_name == "sql_secret":
        return json.dumps(
            {
                "HOST": "localhost",
                "USER": "postgres",
                "PASSWORD": "postgres",
                "PORT": 5432,
                "DB_NAME": "postgres",
            }
        )
    elif secret_name == "mongo_secret":
        return json.dumps(
            {
                "HOST": "localhost",
                "USERNAME": "test_user",
                "PASSWORD": "test_password",
                "PORT": 27017,
                "DATABASE": "test_db",
                "AUTH_SOURCE": "test_db",
            }
        )
    else:
        raise ValueError("Unknown secret name")


@mock.patch("credentials.get_secret")
def test_step_start(
    mock_credentials,
    produce_messages,
    mongo_service,
    env_variables,
    psql_conn,
    mongo_conn,
    kafka_consumer,
):
    from scripts.run_step import step_creator

    mock_credentials.side_effect = mock_get_secret
    produce_messages("correction")
    step_creator().start()

    consumer = kafka_consumer(["lightcurve"])
    messages = list(consumer.consume())
    assert len(messages) == 10
    for msg in messages:
        detections = msg["detections"]
        oids = set(map(lambda x: x["oid"], msg["detections"]))
        assert len(oids) == 1
        if oids.pop() == "ZTF000llmn":
            assert len(detections) == 12  # 10 from the input + 2 from the db
        else:
            assert len(detections) == 10  # 10 from the input only
        candids_are_string = list(map(lambda x: type(x) == str, msg["candid"]))
        assert all(candids_are_string)


def test_step_with_explicit_db_config(
    produce_messages,
    mongo_service,
    psql_conn,
    kafka_consumer,
):
    produce_messages("correction1")
    envcopy = os.environ.copy()
    env_variables_dict = {
        "PRODUCER_SCHEMA_PATH": "../schemas/lightcurve_step/output.avsc",
        "CONSUMER_SCHEMA_PATH": "",
        "METRICS_SCHEMA_PATH": "../schemas/lightcurve_step//metrics.json",
        "SCRIBE_SCHEMA_PATH": "../schemas/scribe.avsc",
        "CONSUMER_SERVER": "localhost:9092",
        "CONSUMER_TOPICS": "correction1",
        "CONSUMER_GROUP_ID": "with_explicit_db_config",
        "METRICS_SERVER": "localhost:9092",
        "PRODUCER_SERVER": "localhost:9092",
        "PRODUCER_TOPIC": "lightcurve1",
        "ENABLE_PARTITION_EOF": "True",
        "CONSUME_MESSAGES": "10",
        "MONGO_HOST": "localhost",
        "MONGO_USERNAME": "test_user",
        "MONGO_PASSWORD": "test_password",
        "MONGO_PORT": "27017",
        "MONGO_DATABASE": "test_db",
        "MONGO_AUTH_SOURCE": "test_db",
        "PSQL_HOST": "localhost",
        "PSQL_USERNAME": "postgres",
        "PSQL_PASSWORD": "postgres",
        "PSQL_PORT": "5432",
        "PSQL_DATABASE": "postgres",
    }
    for key in env_variables_dict:
        os.environ[key] = env_variables_dict[key]

    from scripts.run_step import step_creator

    step_creator().start()

    consumer = kafka_consumer(["lightcurve1"])
    assert len(list(consumer.consume())) == 10
    os.environ = envcopy
