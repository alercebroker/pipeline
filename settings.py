import os

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
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'python_example_group_1'
    },
    "TOPICS": ["commands"],
    "NUM_MESSAGES": 2
}

DB_CONFIG = {
    "MONGO": {
        "HOST": os.getenv("MONGO_HOST", "localhost"),
        "USERNAME": os.getenv("MONGO_USER", "mongo"),
        "PASSWORD": os.getenv("MONGO_PASSWORD", "mongo"),
        "PORT": int(os.getenv("MONGO_PORT", 27017)),
        "DATABASE": os.getenv("MONGO_NAME", "test"),
    }
}
## Step Configuration
STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG
    # "N_PROCESS": 4,            # Number of process for multiprocess script
    # "COMMIT": False,           #Disables commit, useful to debug a KafkaConsumer
}