##################################################
#       alert_archiving_step   Settings File
##################################################
import os

## Set the global logging level to debug
#LOGGING_DEBUG = True

## Consumer configuration
### Each consumer has different parameters and can be found in the documentation
CONSUMER_CONFIG = {
    "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset":"beginning",
        "max.poll.interval.ms": 3600000
    },
    "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 60)),
    "consume.messages": int(os.getenv("CONSUME_MESSAGES", 2200)),
}

STEP_METADATA = {
    "STEP_VERSION": os.getenv("STEP_VERSION", "dev"),
    "STEP_ID": os.getenv("STEP_ID", "preprocess"),
    "STEP_NAME": os.getenv("STEP_NAME", "preprocess"),
    "STEP_COMMENTS": os.getenv("STEP_COMMENTS", ""),
}



## Step Configuration
STEP_CONFIG = {
    "N_PROCESS": os.getenv("N_PROCESS"),            # Number of process for multiprocess script
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "STEP_METADATA": STEP_METADATA,
}