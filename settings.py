##################################################
#       features   Settings File
##################################################
import os

CONSUMER_CONFIG = {
    "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"]
    }
}

DB_CONFIG = {
    "PSQL": {
        "HOST": os.environ["DB_HOST"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "PORT": int(os.environ["DB_PORT"]),
        "DB_NAME": os.environ["DB_NAME"]
    }
}

ES_CONFIG = {
    "INDEX_PREFIX": os.environ["ES_PREFIX"],
    "host": os.environ["ES_NETWORK_HOST"],
    "port": os.environ["ES_NETWORK_PORT"]
}

PRODUCER_CONFIG = {
    "TOPIC": os.environ["PRODUCER_TOPIC"],
    "PARAMS": {
        'bootstrap.servers': os.environ["PRODUCER_SERVER"],
    },
    "SCHEMA": {
        'doc': 'Late Classification',
        'name': 'probabilities + features',
        'type': 'record',
        'fields': [
            {'name': 'candid', 'type': 'string'},
            {'name': 'oid', 'type': 'string'},
            {
                'name': 'features',
                'type': {
                    'type': 'map',
                    'values': ['float', 'int', 'string', 'null']
                }
            },
            {
                'name': 'late_classification',
                'type': {
                    'type': 'record',
                    'name': 'late_record',
                    'fields': [
                        {
                            'name': 'probabilities',
                            'type': {
                                'type': 'map',
                                'values': ['float'],
                            }
                        },
                        {
                            'name': 'class',
                            'type': 'string'
                        },
                        {
                            'name': 'hierarchical',
                            'type':
                            {
                                'name': 'root',
                                'type': 'map',
                                'values': [
                                    {
                                        'type': 'map',
                                        'values': 'float'
                                    },
                                    {
                                        'type': 'map',
                                        'values': {
                                            'type': 'map',
                                            'values': 'float'
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        ]
    }
}

STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "ES_CONFIG": ES_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
    "CLASSIFIER_NAME": os.environ["CLASSIFIER_NAME"],
    "TAXONOMY_NAME": os.environ["TAXONOMY_NAME"],
    "COMMIT": False
}
