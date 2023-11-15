import os
import pathlib
from credentials import get_credentials

##################################################
#       mongo_scribe   Settings File
##################################################

# SCHEMA PATH RELATIVE TO THE SETTINGS FILE
producer_schema_path = pathlib.Path(
    pathlib.Path(__file__).parent.parent, "schemas/scribe_step", "output.avsc"
)
metrics_schema_path = pathlib.Path(
    pathlib.Path(__file__).parent.parent, "schemas/scribe_step", "metrics.json"
)
scribe_schema_path = pathlib.Path(
    pathlib.Path(__file__).parent.parent, "schemas/scribe_step", "scribe.avsc"
)

## Consumer configuration
### Each consumer has different parameters and can be found in the documentation
CONSUMER_CONFIG = {
    "CLASS": os.getenv("CONSUMER_CLASS", "apf.consumers.KafkaConsumer"),
    "PARAMS": {
        "bootstrap.servers": os.environ["CONSUMER_SERVER"],
        "group.id": os.environ["CONSUMER_GROUP_ID"],
        "auto.offset.reset": "beginning",
    },
    # "TOPICS": ["w_Object", "w_Detections", "w_Non_Detections"],
    "TOPICS": os.environ["TOPICS"].strip().split(","),
    "NUM_MESSAGES": int(os.getenv("NUM_MESSAGES", "50")),
    "TIMEOUT": int(os.getenv("TIMEOUT", "10")),
}

if os.getenv("KAFKA_USERNAME") and os.getenv("KAFKA_PASSWORD"):
    CONSUMER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
    CONSUMER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    CONSUMER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("KAFKA_USERNAME")
    CONSUMER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("KAFKA_PASSWORD")

db_type = os.getenv("DB_ENGINE", "mongo")

DB_CONFIG = {}
if db_type == "mongo":
    DB_CONFIG["MONGO"] = get_credentials(os.environ["DB_SECRET_NAME"], db_type)
elif db_type == "sql":
    DB_CONFIG["PSQL"] = get_credentials(os.environ["DB_SECRET_NAME"], db_type)

METRICS_CONFIG = {
    "CLASS": "apf.metrics.KafkaMetricsProducer",
    "PARAMS": {
        "PARAMS": {
            "bootstrap.servers": os.environ["METRICS_HOST"],
        },
        "TOPIC": os.environ["METRICS_TOPIC"],
        "SCHEMA_PATH": os.getenv("METRICS_SCHEMA_PATH", metrics_schema_path),
    },
}

if os.getenv("METRICS_KAFKA_USERNAME") and os.getenv("METRICS_KAFKA_PASSWORD"):
    METRICS_CONFIG["PARAMS"]["PARAMS"]["security.protocol"] = "SASL_SSL"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.username"] = os.getenv(
        "METRICS_KAFKA_USERNAME"
    )
    METRICS_CONFIG["PARAMS"]["PARAMS"]["sasl.password"] = os.getenv(
        "METRICS_KAFKA_PASSWORD"
    )

## Step Configuration
STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "DB_TYPE": db_type,
    "CONSUMER_CONFIG": CONSUMER_CONFIG,
    "METRICS_CONFIG": METRICS_CONFIG,
    "PROMETHEUS": bool(os.getenv("USE_PROMETHEUS", "True")),
    "RETRIES": int(os.getenv("RETRIES", "3")),
    "RETRY_INTERVAL": int(os.getenv("RETRY_INTERVAL", "1")),
    "USE_PROFILING": bool(os.getenv("USE_PROFILING", True)),
    "PYROSCOPE_SERVER": os.getenv(
        "PYROSCOPE_SERVER", "http://pyroscope.pyroscope:4040"
    ),
}
