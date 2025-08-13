# This script fetches a couple messages from an LSST Kafka topic
# and save them as avro files for use in tests.

from confluent_kafka import Consumer
import os
from apf.core.settings import config_from_yaml_file


kafka_config = config_from_yaml_file(os.environ["CONFIG_YAML_PATH"])["CONSUMER_CONFIG"]
consumer = Consumer(kafka_config["PARAMS"])
consumer.subscribe(kafka_config["TOPICS"])

messages = []
while len(messages) < 5:
    msg_batch = consumer.consume(num_messages=1, timeout=5)
    for message in msg_batch:
        if message.error():
            print(f"Error in kafka stream: {message.error()}")
            continue
        else:
            messages.append(message.value())

if not os.path.exists("avro_messages"):
    os.makedirs("avro_messages")

for i, message in enumerate(messages):
    with open(f"avro_messages/message_{i}.avro", "wb") as f:
        f.write(message)
