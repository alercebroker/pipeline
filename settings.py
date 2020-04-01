##################################################
#       correction   Settings File
##################################################
import os
LOGGING_DEBUG = True
DB_CONFIG = {
    "PSQL": {
        "HOST": os.environ["DB_HOST"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "PORT": int(os.environ["DB_PORT"]),
        "DB_NAME": os.environ["DB_NAME"]
    }
}

CONSUMER_CONFIG = {
    "DIRECTORY_PATH": "/test_pipeline/test_data/tests/ztf_sample/"
}


PRODUCER_CONFIG = {
    "TOPIC": os.environ["PRODUCER_TOPIC"],
    "PARAMS": {
        'bootstrap.servers': os.environ["PRODUCER_SERVER"],
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
                         }}
                    ]
                }
            }},
            {'name': 'non_detections', 'type': {
                'type': 'array',
                'items': {
                    'type': 'map',
                    'values': ['float', 'int', 'string', 'null']
                }
            }}
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
    # "ES_CONFIG": ES_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "COMMIT": False,
}
