import os
from random import choice

from apf.producers.kafka import KafkaProducer
from db_plugins.db.mongo._connection import MongoConnection
from mongo_scribe.step import MongoScribe

from _generator import CommandGenerator

DB_CONFIG = {
    "MONGO": {
        "host": "localhost",
        "username": "",
        "password": "",
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
    "SCHEMA_PATH": os.path.join(
        os.path.dirname(__file__), "producer_schema.avsc"
    ),
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

db = MongoConnection(DB_CONFIG["MONGO"])
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
    [
        generator.generate_random_command({"upsert": True}, 125)
        for _ in range(125)
    ]
)
# 1000 - 1000 + X (upsert and set_on_insert)
generator.set_offset(250)
commands.append(generator.generate_insert())
commands.extend(
    [
        generator.generate_random_command(
            {"upsert": True, "set_on_insert": True}, 250
        )
        for _ in range(125)
    ]
)
# ask for possible edge cases


def test_bulk(kafka_service, mongo_service):
    db.create_db()

    for i, command in enumerate(commands):
        producer.produce(command)

    producer.producer.flush(10)
    step.start()
    collection = step.db_client.connection.database["object"]

    # get any element that have features (obtained from the tracker)
    updated_feats = {
        key: val
        for key, val in generator.get_updated_features().items()
        if val != []
    }
    sample_id = choice(list(updated_feats.keys()))
    result = collection.find_one({"_id": f"ID{sample_id}"})
    tracked = updated_feats[sample_id]
    assert result["features"] == tracked

    # # the same as above but with probabilities
    updated_probs = {
        key: val
        for key, val in generator.get_updated_probabilities().items()
        if val != []
    }
    sample_id_2 = choice(list(updated_probs.keys()))
    result = collection.find_one({"_id": f"ID{sample_id_2}"})
    tracked = updated_probs[sample_id_2]

    assert len(result["probabilities"]) == len(tracked)
    for probs in result["probabilities"]:
        probs["values"].sort(key=lambda x: x["ranking"])

    for probs in tracked:
        probs["values"].sort(key=lambda x: x["ranking"])

    assert result["probabilities"] == tracked

    # assertIsNotNone(result)

    # check edge cases
