##################################################
#       correction   Settings File
##################################################

CONSUMER_CONFIG = {
    "DIRECTORY_PATH": "ztf_public_20190302"
}
STEP_CONFIG = {
    "DB_CONFIG": {
        "PSQL": {
            "HOST": "localhost",
            "USER": "postgres",
            "PASSWORD": "docker",
            "PORT": 5432,
            "DB_NAME": "test"
        }
    },
    "PRODUCER_CONFIG": {
        "TOPIC": "test_offset",
        "PARAMS": {
            'bootstrap.servers': '127.0.0.1:9092',
        },
        "SCHEMA": {
            'doc': 'Lightcurve',
            'name': 'lightcurve',
            'type': 'record',
            'fields': [
                {'name': 'oid', 'type': 'string'},
                {'name': 'detections', 'type': {
                    'type': 'array',
                    'items': {'type': 'map', 'values': ['float', 'int', 'string', {"type": "map", 'values': ['float', 'int', 'string']}]}
                }},
                {'name': 'non_detections', 'type': {
                    'type': 'array',
                    'items': {'type': 'map', 'values': ['float', 'int', 'string']}
                }}
            ],
        }
    }
}
DB_CONFIG = {
    "PSQL": {
        "HOST": "localhost",
        "USER": "postgres",
        "PASSWORD": "docker",
        "PORT": 5432,
        "DB_NAME": "test"
    }
}
