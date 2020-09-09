##################################################
#       features   Settings File
##################################################
import os
from features_schema import FEATURES_SCHEMA

CONSUMER_CONFIG = {
    "CLASS": os.getenv("CONSUMER_CLASS", "apf.consumers.KafkaConsumer"),
    "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"]
    }
}

DB_CONFIG = {
    "SQL": {
        "ENGINE": os.getenv("DB_ENGINE", "postgresql"),
        "HOST": os.environ["DB_HOST"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "PORT": int(os.getenv("DB_PORT",5432)),
        "DB_NAME": os.environ["DB_NAME"]
    }
}

PRODUCER_CONFIG = {
    "TOPIC": os.environ["PRODUCER_TOPIC"],
    "PARAMS": {
        'bootstrap.servers': os.environ["PRODUCER_SERVER"],
    },
    "SCHEMA": {
        'doc': 'Late Classification',
        'name': 'probabilities_and_features',
        'type': 'record',
        'fields': [
            {'name': 'oid', 'type': 'string'},
            FEATURES_SCHEMA,
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
    "PRODUCER_CONFIG": PRODUCER_CONFIG,
}
