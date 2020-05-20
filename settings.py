##################################################
#       s3_step   Settings File
##################################################
import os

ES_CONFIG = {
    "INDEX_PREFIX": os.environ["ES_PREFIX"],
    "host": os.environ["ES_NETWORK_HOST"],
    "port": os.environ["ES_NETWORK_PORT"]
}

# Other parameters that can be passed are defined here
# https://elasticsearch-py.readthedocs.io/en/master/api.html#elasticsearch.Elasticsearch

# Consumer configuration
# Each consumer has different parameters and can be found in the documentation-
CONSUMER_CONFIG = {
    "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
    "PARAMS": {
         "bootstrap.servers": os.environ["CONSUMER_SERVER"],
         "group.id": os.environ["CONSUMER_GROUP_ID"]
    }
}

# Database configuration
# Depending on the database backend the parameters can change
DB_CONFIG = {}

# https://stackoverflow.com/questions/45981950/how-to-specify-credentials-when-connecting-to-boto3-s3
STORAGE_CONFIG = {
    "BUCKET_NAME": os.environ["BUCKET_NAME"]
}

PRODUCER_CONFIG = {
    "TOPIC": os.getenv("PRODUCER_TOPIC", None),
    "PARAMS": {
        'bootstrap.servers': os.getenv("PRODUCER_SERVER", None),
    },
    "SCHEMA": {
            'doc': 'S3 file',
            'name': 'S3 step',
            'type': 'record',
            'fields': [
                {'name': 'candid', 'type': 'string'}
        ]
    }
}

# Step Configuration
STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "ES_CONFIG": ES_CONFIG,
    "STORAGE": STORAGE_CONFIG,
    "PRODUCER_CONFIG": PRODUCER_CONFIG
}

