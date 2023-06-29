import os

##################################################
#       cmirrormaker   Settings File
##################################################

# Set the global logging level to debug
LOGGING_DEBUG = os.getenv('LOGGING_DEBUG', False)

# Consumer configuration
# Each consumer has different parameters and can be found in the documentation
CONSUMER_CONFIG = {
    'CLASS': 'cmirrormaker.utils.RawKafkaConsumer',
    'PARAMS': {
        'bootstrap.servers': os.environ['CONSUMER_SERVER'],
        'group.id': os.environ['CONSUMER_GROUP_ID'],
        'auto.offset.reset': 'beginning',
        'max.poll.interval.ms': 3600000
    },
    'consume.timeout': int(os.getenv('CONSUME_TIMEOUT', 10)),
    'consume.messages': int(os.getenv('CONSUME_MESSAGES', 1000))
}

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
    'CLASS': 'cmirrormaker.utils.RawKafkaProducer',
    'TOPIC': os.getenv('PRODUCER_TOPIC'),
    'PARAMS': {
        'bootstrap.servers': os.environ['PRODUCER_SERVER'],
        'acks': 'all',
    }
}

METRICS_CONFIG = {
    'CLASS': 'apf.metrics.KafkaMetricsProducer',
    'EXTRA_METRICS': [],  # This must be kept empty
    'PARAMS': {
        'PARAMS': {
            'bootstrap.servers': os.environ['METRICS_HOST'],
        },
        'TOPIC': os.environ['METRICS_TOPIC'],
        'SCHEMA': {
            '$schema': 'http://json-schema.org/draft-07/schema',
            '$id': 'http://example.com/example.json',
            'type': 'object',
            'title': 'ALeRCE reflector metrics schema',
            'description': 'Metrics for custom mirrormaker used in ALeRCE pipeline.',
            'default': {},
            'examples': [
                {
                    'timestamp_sent': '2020-09-01',
                    'timestamp_received': '2020-09-01',
                }
            ],
            'required': ['timestamp_sent', 'timestamp_received'],
            'properties': {
                'timestamp_sent': {
                    '$id': '#/properties/timestamp_sent',
                    'type': 'string',
                    'title': 'The timestamp_sent schema',
                    'description': 'Timestamp sent refers to the time at which a message is sent.',
                    'default': '',
                    'examples': ['2020-09-01'],
                },
                'timestamp_received': {
                    '$id': '#/properties/timestamp_received',
                    'type': 'string',
                    'title': 'The timestamp_received schema',
                    'description': 'Timestamp received refers to the time at which a message is received.',
                    'default': '',
                    'examples': ['2020-09-01'],
                }
            }
        }
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
