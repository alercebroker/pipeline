##################################################
#       correction   Settings File
##################################################

CONSUMER_CONFIG = {
    "DIRECTORY_PATH": "/home/tronco/Desktop/alerce/new_pipeline/correction_step/tests/ztf_sample"
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

ES_CONFIG = {"INDEX_PREFIX":"ztf_pipeline"}

STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "ES_CONFIG": ES_CONFIG,
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
                    'items': {
                        'type': 'record',
                        'name': 'detection',
                        'fields': [
                            {'name':'candid', 'type':'string'},
                            {'name':'mjd', 'type':'float'},
                            {'name':'fid', 'type':'int'},
                            {'name':'magpsf', 'type':['float','null'],'default':None},
                            {'name':'magap', 'type':['float','null'],'default':None},
                            {'name':'sigmapsf', 'type':['float','null'],'default':None},
                            {'name':'sigmgap', 'type':['float','null'],'default':None},
                            {'name':'ra', 'type':'float'},
                            {'name':'dec', 'type':'float'},
                            {'name':'rb', 'type':['float','null'],'default':None},
                            {'name':'oid', 'type':'string'},
                            { 'name':'alert',
                              'type':{
                                'type':'map',
                                'values': ['int','float','string','null']
                                }
                             }
                        ]
                    }
                }},
                {'name': 'non_detections', 'type': {
                    'type': 'array',
                    'items': {'type': 'map', 'values': ['float', 'int', 'string','null']}
                }}
            ],
        }
    }
}
