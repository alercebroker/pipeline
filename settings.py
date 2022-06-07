import os

##################################################
#       cmirrormaker   Settings File
##################################################

# Set the global logging level to debug
LOGGING_DEBUG = os.getenv('LOGGING_DEBUG', False)

# Consumer configuration
# Each consumer has different parameters and can be found in the documentation
CONSUMER_CONFIG = {
    'CLASS': 'cmirrormaker.utils.CustomKafkaConsumer',
    'PARAMS': {
        'bootstrap.servers': os.environ['CONSUMER_SERVER'],
        'group.id': os.environ['CONSUMER_GROUP_ID'],
        'auto.offset.reset': 'beginning'
    },
    'consume.timeout': int(os.getenv('CONSUME_TIMEOUT', 10)),
    'consume.messages': int(os.getenv('CONSUME_MESSAGES', 1000))
}

PRODUCER_CONFIG = {
    'CLASS': 'cmirrormaker.utils.CustomKafkaProducer',
    'TOPIC': os.environ['PRODUCER_TOPIC'],
    'PARAMS': {
        'bootstrap.servers': os.environ['PRODUCER_SERVER']
    }
}

METRICS_CONFIG = {
    'CLASS': 'apf.metrics.KafkaMetricsProducer',
    'PARAMS': {
        'PARAMS': {
            'bootstrap.servers': os.environ['METRICS_HOST'],
            'auto.offset.reset': 'smallest'
        },
        'TOPIC': os.environ['METRICS_TOPIC']
    }
}

# Step Configuration
STEP_CONFIG = {
    'CONSUMER_CONFIG': CONSUMER_CONFIG,
    'PRODUCER_CONFIG': PRODUCER_CONFIG,
    'METRICS_CONFIG': METRICS_CONFIG,
    "N_PROCESS": os.getenv('N_PROCESS')
}
