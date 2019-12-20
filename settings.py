##################################################
#       correction   Settings File
##################################################

CONSUMER_CONFIG = {
    "DIRECTORY_PATH": "ztf_public_20190302"
    # "TOPICS": ["mag_test"],
    # "PARAMS": {
    #     "bootstrap.servers": "kafka1.alerce.online:9092",
    #     "group.id": "test"
    # }
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

# ES_CONFIG = {"INDEX_PREFIX":"ztf_pipeline"}

STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    # "ES_CONFIG": ES_CONFIG,
    "PRODUCER_CONFIG": {
        "TOPIC": "mag_test",
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
                    'items': {
                        'type': 'record',
                        'name': 'detection',
                        'fields': [
                            {'name': 'candid', 'type': 'string'},
                            {'name': 'mjd', 'type': 'float'},
                            {'name': 'fid', 'type': 'int'},
                            {'name': 'magpsf_corr', 'type': [
                                'float', 'null'], 'default':None},
                            {'name': 'magap_corr', 'type': [
                                'float', 'null'], 'default':None},
                            {'name': 'sigmapsf_corr', 'type': [
                                'float', 'null'], 'default':None},
                            {'name': 'sigmagap_corr', 'type': [
                                'float', 'null'], 'default':None},
                            {'name': 'ra', 'type': 'float'},
                            {'name': 'dec', 'type': 'float'},
                            {'name': 'rb', 'type': [
                                'float', 'null'], 'default':None},
                            {'name': 'oid', 'type': 'string'},
                            {'name': 'alert',
                             'type': {
                                 'type': 'map',
                                 'values': ['int', 'float', 'string', 'null']
                             }
                             }
                        ]
                    }
                }},
                {'name': 'non_detections', 'type': {
                    'type': 'array',
                    'items': {'type': 'map', 'values': ['float', 'int', 'string', 'null']}
                }}
            ],
        }
    },
    "STORAGE":{
        "NAME": "ztf-storage"
    }
}
