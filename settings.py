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
STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "ES_CONFIG" : ES_CONFIG,
    "FEATURE_VERSION": "v0.1"
}
