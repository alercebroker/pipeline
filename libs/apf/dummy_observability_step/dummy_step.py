import logging
import json
from apf.core.step import GenericStep
import sys


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "level": record.levelname,
            "timestamp": self.formatTime(record, self.datefmt),
            "step": record.name,
            "message": record.getMessage(),
            "survey": record.survey,
        }
        # Add extra fields if present
        if hasattr(record, "n_messages"):
            log_record["n_messages"] = record.n_messages
        # Add exception info if present
        if record.exc_info:
            log_record["traceback"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


class SurveyFilter(logging.Filter):
    def __init__(self, survey):
        super().__init__()
        self.survey = survey

    def filter(self, record):
        record.survey = self.survey
        return True


class DummyStep(GenericStep):
    def __init__(self, config: dict, level=logging.INFO, **step_args):
        super().__init__(config=config, level=level, **step_args)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        file_handler = logging.FileHandler("dummy_step.log")
        file_handler.setFormatter(JSONFormatter())
        self.logger.handlers = []  # Remove inherited handlers
        self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)
        self.logger.addFilter(SurveyFilter(self.survey))  # Add the filter here

    def execute(self, messages: list) -> list:
        # Example log with context
        self.logger.info("Processing batch")
        # self.logger.debug("dummy debug message")
        # self.logger.warning("dummy warning message")
        # self.logger.error("dummy error message")
        # self.logger.critical("dummy critical message")
        return messages

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
