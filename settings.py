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

# Step Configuration
STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "ES_CONFIG": ES_CONFIG,
    "STORAGE": STORAGE_CONFIG,
    # "N_PROCESS": 4,            # Number of process for multiprocess script
    "COMMIT": False,  # Disables commit, useful to debug KafkaConsumer
}

