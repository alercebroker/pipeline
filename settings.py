import os
import json

##################################################
#       simulator   Settings File
##################################################

# Set the global logging level to debug
# LOGGING_DEBUG = True

# Consumer configuration
CONSUMER_CONFIG = {
    "CLASS": os.getenv("CONSUMER_CLASS", "apf.consumers.AVROFileConsumer"),
    "DIRECTORY_PATH": "data/",
    "consume.messages": int(os.getenv("MESSAGES", "1000")),
}

# Producer configuration
SCHEMA_PATH = os.environ["SCHEMA_PATH"]

with open(SCHEMA_PATH) as f:
    SCHEMA = json.load(f)
PRODUCER_CONFIG = {
    "CLASS": os.getenv("PRODUCER_CLASS", "apf.producers.KafkaProducer"),
    "TOPIC": os.environ["PRODUCER_TOPIC"],
    "PARAMS": {
        "bootstrap.servers": os.environ["PRODUCER_SERVER"],
    },
    "SCHEMA": SCHEMA,
}

if os.getenv("KAFKA_USERNAME") and os.getenv("KAFKA_PASSWARD"):
    PRODUCER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("KAFKA_USERNAME")
    PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("KAFKA_PASSWORD")

# Step Configuration
STEP_CONFIG = {
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "MESSAGES": int(os.getenv("MESSAGES", "100")),
    "EXPOSURE_TIME": int(os.getenv("EXPOSURE_TIME", "1")),
    "PROCESS_TIME": int(os.getenv("PROCESS_TIME", "1")),
}
