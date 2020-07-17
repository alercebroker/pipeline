##################################################
#       Correction   Settings File
##################################################
import os
DB_CONFIG = {
    "SQL": {
        "ENGINE": os.environ["DB_ENGINE"],
        "HOST": os.environ["DB_HOST"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "PORT": int(os.environ["DB_PORT"]),
        "DB_NAME": os.environ["DB_NAME"]
    }
}

CONSUMER_CONFIG = {
    "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
    "PARAMS": {
         "bootstrap.servers": os.environ["CONSUMER_SERVER"],
         "group.id": os.environ["CONSUMER_GROUP_ID"]
    },
    "DIRECTORY_PATH": os.environ["AVRO_PATH"]
}

XMATCH = {
	'type' : 'map',
    'values' : {
        'type' : 'map',
        'values' : [ "string", "float", "null"]
    }
}


PRODUCER_CONFIG = {
    "TOPIC": os.environ["PRODUCER_TOPIC"],
    "PARAMS": {
        'bootstrap.servers': os.environ["PRODUCER_SERVER"],
        'message.max.bytes': 6291456
    },
    "SCHEMA": {
        'doc': 'Light curve',
        'name': 'light_curve',
        'type': 'record',
        'fields': [
            {'name': 'oid', 'type': 'string', 'default': None},
            {'name': 'candid', 'type': 'string'},
            {'name': 'fid', 'type': 'int'},
            {'name': 'detections', 'type': {
                'type': 'array',
                'items': {
                    'type': 'map',
                    'values': ['float', 'int', 'string', 'null', 'boolean']
                }
            }},
            {'name': 'non_detections', 'type': {
                'type': 'array',
                'items': {
                    'type': 'map',
                    'values': ['float', 'int', 'string', 'null']
                }
            }},
            {'name': 'xmatches', 'type': [XMATCH, "null"], "default": "null"},
        ],
    }
}
ES_CONFIG = {
    "INDEX_PREFIX": os.environ["ES_PREFIX"],
    "host": os.environ["ES_NETWORK_HOST"],
    "port": os.environ["ES_NETWORK_PORT"]
}

STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "ES_CONFIG": ES_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "N_PROCESS": os.getenv("N_PROCESS"),
    "STEP_VERSION": os.getenv("STEP_VERSION", "dev")
}
