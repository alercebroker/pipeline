##################################################
#       features   Settings File
##################################################



CONSUMER_CONFIG = {"TOPICS": ["feature_test"],
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
    "TOPIC": "late_classification_test",
    "PARAMS": {
        'bootstrap.servers': '127.0.0.1:9092',
    },
    "SCHEMA": {
        'doc': 'Late Classification',
        'name': 'probabilities',
        'type': 'record',
        'fields': [
            {'name': 'oid', 'type': 'string'},
            {'name': 'probabilties', 'type': {
                'type': 'map',
                'values': ['float']
                }
            },
            {'name': 'class', 'type':'string'},
            {'name':'hierarchical', 'type':{
                'type': 'array',
                'items': [{'type': 'map','values':['float', {'type':'map', 'values': 'float'}]}]
            }}
        ]
    }
}

STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "ES_CONFIG" : ES_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "COMMIT": False,
}
