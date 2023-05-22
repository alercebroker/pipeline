import os
from credentials import get_mongodb_credentials

##################################################
#       mongo_scribe   Settings File
##################################################

## Set the global logging level to debug
LOGGING_DEBUG = True

## Consumer configuration
### Each consumer has different parameters and can be found in the documentation
CONSUMER_CONFIG = {
    "CLASS": "apf.consumers.KafkaConsumer",
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "beginning"
    },
    # "TOPICS": ["w_Object", "w_Detections", "w_Non_Detections"],
    "TOPICS": os.environ["TOPICS"].strip().split(","),
    "NUM_MESSAGES": int(os.getenv("NUM_MESSAGES", "50")),
}

if os.getenv("KAFKA_USERNAME") and os.getenv("KAFKA_PASSWORD"):
    CONSUMER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    CONSUMER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    CONSUMER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("KAFKA_USERNAME")
    CONSUMER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("KAFKA_PASSWORD")

DB_CONFIG = {
    "MONGO": get_mongodb_credentials()
}

## Step Configuration
STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG
    # "N_PROCESS": 4,            # Number of process for multiprocess script
    # "COMMIT": False,           #Disables commit, useful to debug a KafkaConsumer
}
