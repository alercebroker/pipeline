##################################################
#       features   Settings File
##################################################

LOGGING_DEBUG=True

CONSUMER_CONFIG = {"TOPICS": ["test_offset"],
                   "PARAMS": {
                        'bootstrap.servers': '127.0.0.1:9092',
                        'group.id': "test1"
                    }
                   }
DB_CONFIG = {
    "PSQL":{
        "HOST": "localhost",
        "USER": "postgres",
        "PASSWORD": "docker",
        "PORT": 5432,
        "DB_NAME": "test"
    }
}
ES_CONFIG = {"INDEX_PREFIX":"ztf_pipeline"}

PRODUCER_CONFIG = {
    "TOPIC": "feature_test",
    "PARAMS": {
        'bootstrap.servers': '127.0.0.1:9092',
    },
    "SCHEMA": {
        'doc': 'Features',
        'name': 'features',
        'type': 'record',
        'fields': [
            {'name': 'oid', 'type': 'string'},
            {'name': 'features', 'type': {
                'type': 'map',
                'values': ['float', 'int', 'string','null']
                }
            }
        ]
    }
}

STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "ES_CONFIG" : ES_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "FEATURE_VERSION": "v0.1"
}
