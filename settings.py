##################################################
#       Correction   Settings File
##################################################
import os
from schema import SCHEMA

DB_CONFIG = {
    "SQL": {
        "ENGINE": os.environ["DB_ENGINE"],
        "HOST": os.environ["DB_HOST"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "PORT": int(os.environ["DB_PORT"]),
        "DB_NAME": os.environ["DB_NAME"],
    }
}

CONSUMER_CONFIG = {
    "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
    },
    "DIRECTORY_PATH": os.environ["AVRO_PATH"],
}


PRODUCER_CONFIG = {
    "TOPIC": os.environ["PRODUCER_TOPIC"],
    "PARAMS": {
        "bootstrap.servers": os.environ["PRODUCER_SERVER"],
        "message.max.bytes": 6291456,
    },
    "SCHEMA": SCHEMA,
}

STEP_METADATA = {
    "STEP_VERSION": os.getenv("STEP_VERSION", "dev"),
    "STEP_ID": os.getenv("STEP_ID", "preprocess"),
    "STEP_NAME": os.getenv("STEP_NAME", "preprocess"),
    "STEP_COMMENTS": os.getenv("STEP_COMMENTS", ""),
}

STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "N_PROCESS": os.getenv("N_PROCESS"),
    "STEP_METADATA": STEP_METADATA,
}
