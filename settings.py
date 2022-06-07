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

# TODO: Check whether both possibilities can apply or only one should be used
if os.getenv('TOPIC_STRATEGY_FORMAT'):
    CONSUMER_CONFIG['TOPIC_STRATEGY'] = {
        'CLASS': 'apf.core.topic_management.DailyTopicStrategy',
        'PARAMS': {
            'topic_format': os.environ['TOPIC_STRATEGY_FORMAT'].strip().split(','),
            'date_format': '%Y%m%d',
            'change_hour': 23,
        },
    }
elif os.getenv('CONSUMER_TOPICS'):
    CONSUMER_CONFIG['TOPICS'] = os.environ['CONSUMER_TOPICS'].strip().split(',')
else:
    raise Exception('Add TOPIC_STRATEGY or CONSUMER_TOPICS')

PRODUCER_CONFIG = {
    'CLASS': 'cmirrormaker.utils.CustomKafkaProducer',
    'TOPIC': os.environ['PRODUCER_TOPIC'],
    'PARAMS': {
        'bootstrap.servers': os.environ['PRODUCER_SERVER']
    }
}

METRICS_CONFIG = {
    'CLASS': 'apf.metrics.KafkaMetricsProducer',
    'EXTRA_METRICS': [],
    'PARAMS': {
        'PARAMS': {
            'bootstrap.servers': os.environ['METRICS_HOST'],
        },
        'TOPIC': os.environ['METRICS_TOPIC']
    }
}

STEP_METADATA = {
    'STEP_VERSION': os.getenv('STEP_VERSION', 'dev'),
    'STEP_ID': os.getenv('STEP_ID', 'cmirrormaker'),
    'STEP_NAME': os.getenv('STEP_NAME', 'cmirrormaker'),
    'STEP_COMMENTS': os.getenv('STEP_COMMENTS', ''),
}

# Step Configuration
STEP_CONFIG = {
    'CONSUMER_CONFIG': CONSUMER_CONFIG,
    'PRODUCER_CONFIG': PRODUCER_CONFIG,
    'METRICS_CONFIG': METRICS_CONFIG,
    'N_PROCESS': os.getenv('N_PROCESS'),
    'STEP_METADATA': STEP_METADATA
}
