from random import choice
from mongo_scribe.step import MongoScribe
from apf.producers.kafka import KafkaProducer
from _generator import CommandGenerator

DB_CONFIG = {
    "MONGO": {
        "host": "localhost",
        "username": "mongo",
        "password": "mongo",
        "port": 27017,
        "database": "test",
    }
}

CONSUMER_CONFIG = {
    "CLASS": "apf.consumers.KafkaConsumer",
    "TOPICS": ["test_topic_2"],
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
        "group.id": "command_consumer_1",
        "enable.partition.eof": True,
        "auto.offset.reset": "beginning",
    },
    "NUM_MESSAGES": 25,
    "TIMEOUT": 15,
}

PRODUCER_CONFIG = {
    "TOPIC": "test_topic_2",
    "PARAMS": {"bootstrap.servers": "localhost:9092"},
    "SCHEMA": {
        "namespace": "db_operation",
        "type": "record",
        "name": "Command",
        "fields": [
            {"name": "payload", "type": "string"},
        ],
    },
}


step_config = {
    "DB_CONFIG": DB_CONFIG,
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "STEP_METADATA": {
        "STEP_ID": "scribe",
        "STEP_NAME": "scribe",
        "STEP_VERSION": "test",
        "STEP_COMMENTS": "test ver.",
    },
}

step = MongoScribe(config=step_config)
producer = KafkaProducer(config=PRODUCER_CONFIG)
generator = CommandGenerator()

# better generate the full batch then check
# 0 - X (no options)
commands = [generator.generate_insert()]
commands.extend([generator.generate_random_command() for _ in range(125)])
# 500 - 500 + X (with upsert)
generator.set_offset(125)
commands.append(generator.generate_insert())
commands.extend(
    [generator.generate_random_command({"upsert": True}, 125) for _ in range(125)]
)
# 1000 - 1000 + X (upsert and set_on_insert)
generator.set_offset(250)
commands.append(generator.generate_insert())
commands.extend(
    [
        generator.generate_random_command({"upsert": True, "set_on_insert": True}, 250)
        for _ in range(125)
    ]
)
# ask for possible edge cases


def test_bulk(kafka_service, mongo_service):
    for i, command in enumerate(commands):
        producer.produce(command)

    producer.producer.flush(10)
    step.start()
    collection = step.db_client.connection.database["object"]

    # get any element that have features (obtained from the tracker)
    updated_feats = {
        key: val for key, val in generator.get_updated_features().items() if val != {}
    }
    sample_id = choice(list(updated_feats.keys()))
    result = collection.find_one({"_id": f"ID{sample_id}"})
    tracked = updated_feats[sample_id]
    assert result["features"] == tracked

    # # the same as above but with probabilities
    updated_probs = {
        key: val
        for key, val in generator.get_updated_probabilities().items()
        if val != {}
    }
    sample_id_2 = choice(list(updated_probs.keys()))
    result = collection.find_one({"_id": f"ID{sample_id_2}"})
    tracked = updated_probs[sample_id_2]

    classifiers = list(result["probabilities"].keys())
    assert set(classifiers) == set(tracked.keys())
    for classifier in classifiers:
        classifier_vals = result["probabilities"][classifier]
        assert classifier_vals["version"] == tracked[classifier]["version"]
        assert classifier_vals["class_rank_1"] == tracked[classifier]["class_rank_1"]
        assert (
            classifier_vals["probability_rank_1"]
            == tracked[classifier]["probability_rank_1"]
        )
        classifier_vals["values"].sort(key=lambda x: x["ranking"])
        tracked[classifier]["values"].sort(key=lambda x: x["ranking"])
        assert (
           classifier_vals["values"] == tracked[classifier]["values"]
        )

    # assertIsNotNone(result)

    # check edge cases
