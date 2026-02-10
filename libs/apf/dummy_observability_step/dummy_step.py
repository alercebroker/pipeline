import logging
import json
from apf.core.step import GenericStep
import sys


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs as JSON for structured logging.
    
    This makes logs easier to parse, search, and analyze in log aggregation
    systems like OpenSearch/Elasticsearch.
    """
    def format(self, record):
        log_record = {
            "level": record.levelname,
            "timestamp": self.formatTime(record, self.datefmt),
            "step": record.name,
            "message": record.getMessage(),
            "survey": record.survey,
        }
        # Add extra fields if present (passed via logger.info(..., extra={...}))
        if hasattr(record, "n_messages"):
            log_record["n_messages"] = record.n_messages
        # Add exception info if present
        if record.exc_info:
            log_record["traceback"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


class SurveyFilter(logging.Filter):
    """
    Logging filter that adds survey information to every log record.
    
    This ensures all logs from this step include the survey context,
    making it easy to filter logs by survey in Kibana.
    """
    def __init__(self, survey):
        super().__init__()
        self.survey = survey

    def filter(self, record):
        record.survey = self.survey
        return True


class DummyStep(GenericStep):
    """
    Example APF step demonstrating structured JSON logging.
    
    Logs are written to:
    1. stdout (captured by Docker/K8s)
    2. dummy_step.log file (collected by Fluent Bit)
    
    Both outputs use JSON format for easy parsing by log aggregation systems.
    """
    def __init__(self, config: dict, level=logging.INFO, **step_args):
        super().__init__(config=config, level=level, **step_args)
        
        # Setup stdout handler with JSON formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        
        # Setup file handler with JSON formatter (for Fluent Bit collection)
        file_handler = logging.FileHandler("dummy_step.log")
        file_handler.setFormatter(JSONFormatter())
        
        # Replace inherited handlers with our custom ones
        self.logger.handlers = []
        self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)
        
        # Add survey context to all logs
        self.logger.addFilter(SurveyFilter(self.survey))

    def execute(self, messages: list) -> list:
        # Log batch processing with context
        self.logger.info(
            "Processing batch",
            extra={"n_messages": len(messages)}
        )
        
        # Example: processing each message
        processed = []
        for i, msg in enumerate(messages):
            try:
                # Simulate processing
                self.logger.debug(
                    "Processing message",
                    extra={"message_index": i, "id": msg.get("id")}
                )
                
                # Example warning for missing optional data
                if not msg.get("value"):
                    self.logger.warning(
                        "Message missing optional field",
                        extra={"id": msg.get("id"), "field": "value"}
                    )
                
                processed.append(msg)
                
            except Exception as e:
                # Log errors with full context and traceback
                self.logger.error(
                    "Failed to process message",
                    extra={"message_index": i, "id": msg.get("id")},
                    exc_info=True
                )
        
        self.logger.info(
            "Batch processing complete",
            extra={"processed": len(processed), "total": len(messages)}
        )
        
        return processed

    def pre_consume(self):
        # raise NotImplementedError("pre_consume is not implemented in DummyStep")
        pass


if __name__ == "__main__":
    step_config = {
        "LOGGING_LEVEL": "INFO",
        "SURVEY": "LSST",
        "CONSUMER_CONFIG": {
            "CLASS": "apf.consumers.csv.CSVConsumer",
            "FILE_PATH": "dummy_data.csv",
        },
    }

    step = DummyStep(config=step_config, level=step_config["LOGGING_LEVEL"])
    step.start()
