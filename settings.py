##################################################
#       features   Settings File
##################################################
import os
FEATURE_VERSION = "v0.1"

CONSUMER_CONFIG = {"TOPICS": [os.getenv("CONSUMER_TOPICS")],
                   "PARAMS": {
    'bootstrap.servers': os.getenv("CONSUMER_SERVER"),
    'group.id': os.getenv("CONSUMER_GROUP_ID")
}
}
DB_CONFIG = {
    "PSQL": {
        "HOST": os.getenv("DB_HOST"),
        "USER": os.getenv("DB_USER"),
        "PASSWORD": os.getenv("DB_PASSWORD"),
        "PORT": os.getenv("DB_PORT"),
        "DB_NAME": os.getenv("DB_NAME")
    }
}
ES_CONFIG = {"INDEX_PREFIX": os.getenv("ES_PREFIX"),
             "host": os.getenv("ES_NETWORK_HOST"),
             "port": os.getenv("ES_NETWORK_PORT")
             }

PRODUCER_CONFIG = {
    "TOPIC": os.getenv("PRODUCER_TOPIC"),
    "PARAMS": {
        'bootstrap.servers': os.getenv("PRODUCER_SERVER"),
    },
    "SCHEMA": {
        'doc': 'Features',
        'name': 'features',
        'type': 'record',
        'fields': [
            {'name': 'oid', 'type': 'string'},
            {'name': 'features', 'type': {
                'type': 'map',
                'values': ['float', 'int', 'string', 'null']
            }
            }
        ]
    }
}

STEP_CONFIG = {
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "DB_CONFIG": DB_CONFIG,
    "ES_CONFIG": ES_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "FEATURE_VERSION": FEATURE_VERSION,
    "COMMIT": False
}
