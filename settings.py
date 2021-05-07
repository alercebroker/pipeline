##################################################
#       watchlist_step   Settings File
##################################################
import os

## Set the global logging level to debug
# LOGGING_DEBUG = True

## Consumer configuration
### Each consumer has different parameters and can be found in the documentation
CONSUMER_CONFIG = {
    "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "beginning",
        "max.poll.interval.ms": 3600000,
        "enable.partition.eof": os.getenv("ENABLE_PARTITION_EOF", False),
    },
    "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
    "consume.messages": int(os.getenv("CONSUME_MESSAGES", 1000)),
}
## Step Configuration
STEP_CONFIG = {
    "alert_db_config": {
        "SQL": {
            "ENGINE": "postgresql",
            "HOST": os.environ["ALERTS_DB_HOST"],
            "USER": os.environ["ALERTS_DB_USER"],
            "PASSWORD": os.environ["ALERTS_DB_PASSWORD"],
            "PORT": 5432,  # postgresql tipically runs on port 5432. Notice that we use an int here.
            "DB_NAME": os.environ["ALERTS_DB_NAME"],
        },
    },
    "users_db_config": {
        "SQL": {
            "ENGINE": "postgresql",
            "HOST": os.environ["USERS_DB_HOST"],
            "USER": os.environ["USERS_DB_USER"],
            "PASSWORD": os.environ["USERS_DB_PASSWORD"],
            "PORT": 5432,  # postgresql tipically runs on port 5432. Notice that we use an int here.
            "DB_NAME": os.environ["USERS_DB_NAME"],
        }
    },
}
