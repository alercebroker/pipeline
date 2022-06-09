import os
import json

##################################################
#       simulator   Settings File
##################################################

# Set the global logging level to debug
# LOGGING_DEBUG = True

# Consumer configuration
CONSUMER_CONFIG = {
    "CLASS": "apf.consumers.AVROFileConsumer",
    "DIRECTORY_PATH": "data/",
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
        # 'security.protocol': 'SASL_PLAINTEXT',
        # 'sasl.mechanism': 'PLAIN',
        # "sasl.username": "alerce",
        # "sasl.password": "***REMOVED***"
    },
    "SCHEMA": SCHEMA,
}

# Step Configuration
STEP_CONFIG = {
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "MESSAGES": os.getenv("MESSAGES", 100),
    "EXPOSURE_TIME": os.getenv("EXPOSURE_TIME", 1),
    "PROCESS_TIME": os.getenv("PROCESS_TIME", 1),
    "KEY": os.getenv("KEY", "objectId"),
    "N_THREADS": os.getenv("N_THREADS", 1),
}
