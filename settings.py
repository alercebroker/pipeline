import os
##################################################
#       xmatch_step   Settings File
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
CONSUMER_CONFIG = {
			"TOPICS" : ["historic-data"],
			"PARAMS" : {
					"bootstrap.servers": os.environ["KAFKA_SERVER"],
					"group.id" : os.environ["CONSUMER_GROUP"],
			},
			"consume.timeout" : 10,
			"consume.messages" : 1000
}

## Producer Configuration
RESULT_TYPE = {
	'type' : 'map',
	'values' : {
		'type' : 'map',
		'values' : [ "string", "float", "null"]
	}
}

PRODUCER_CONFIG = {
 			"TOPIC": 'xmatch_test',
    		"PARAMS": {
        		'bootstrap.servers': os.environ["KAFKA_SERVER"] ,
        		'message.max.bytes': 6291456
    		},
			"SCHEMA" : {

				'doc' : 'Xmatch',
				'name' : 'xmatch',
				'type' : 'record',
				'fields' : [
					{'name' : 'oid', 'type' : 'string'},
					{'name' : 'result', 'type' : [ "null", RESULT_TYPE ] }
				]
			}
}

## Xmatch Configuration

XMATCH_CONFIG = {
			"CATALOG" : { 
						"name" : "allwise",
						"columns" : [
								'AllWISE',
								'RAJ2000',
								'DEJ2000',
								'W1mag',
								'W2mag',
								'W3mag',
								'W4mag',
								'e_W1mag',
								'e_W2mag',
								'e_W3mag',
								'e_W4mag',
								'Jmag',
								'e_Jmag',
								'Hmag',
								'e_Hmag',
								'Kmag',
								'e_Kmag'
						]
			}
}


## Database configuration
### Depending on the database backend the parameters can change
DB_CONFIG = {}

## Step Configuration
STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
	"PRODUCER_CONFIG" : PRODUCER_CONFIG,
	"XMATCH_CONFIG" : XMATCH_CONFIG
    # "ES_CONFIG": ES_CONFIG,    #Enables metrics for step
    # "N_PROCESS": 4,            # Number of process for multiprocess script
    # "COMMIT": False,           #Disables commit, useful to debug KafkaConsumer
}
