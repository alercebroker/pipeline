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

## Consumer configuration
### Each consumer has different parameters and can be found in the documentation-
CONSUMER_CONFIG = {}

## Database configuration
### Depending on the database backend the parameters can change
DB_CONFIG = {}

## Step Configuration
STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    # "ES_CONFIG": ES_CONFIG,    #Enables metrics for step
    # "N_PROCESS": 4,            # Number of process for multiprocess script
    # "COMMIT": False,           #Disables commit, useful to debug KafkaConsumer
}
