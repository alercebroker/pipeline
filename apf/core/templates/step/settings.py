##################################################
#       {{step_name}}   Settings File
##################################################

## Set the global logging level to debug
#LOGGING_DEBUG = True

## Elasticsearch Metrics Consfiguration
# ES_CONFIG = {
#     "INDEX_PREFIX": "",
#     # Used to generate index index_prefix+class_name+date
#     # Other parameters
# }
## Other parameters that can be passed are defined here
# https://elasticsearch-py.readthedocs.io/en/master/api.html#elasticsearch.Elasticsearch


CONSUMER_CONFIG = {}
DB_CONFIG = {}
STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    # "COMMIT": False,           #Disables commit, useful to debug KafkaConsumer
    # "ES_CONFIG": ES_CONFIG,    #Enables metrics for step
}
