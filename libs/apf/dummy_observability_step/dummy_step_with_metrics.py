import logging
import json
import sys
import time
from apf.core.step import GenericStep
from prometheus_client import Counter, Histogram, Gauge, start_http_server


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
        for field in ["n_messages", "processing_time_ms", "message_index", "id", "field"]:
            if hasattr(record, field):
                log_record[field] = getattr(record, field)
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


class DummyStepWithMetrics(GenericStep):
    """
    Example step demonstrating both structured logging and Prometheus metrics.
    
    Logs are written as JSON to stdout and file for collection by Fluent Bit.
    Metrics are exposed on port 8000 for scraping by Prometheus.
    """
    
    def __init__(self, config: dict, level=logging.INFO, **step_args):
        super().__init__(config=config, level=level, **step_args)
        
        # Setup JSON logging
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        file_handler = logging.FileHandler("dummy_step.log")
        file_handler.setFormatter(JSONFormatter())
        self.logger.handlers = []  # Remove inherited handlers
        self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)
        self.logger.addFilter(SurveyFilter(self.survey))
        
        # Start Prometheus metrics HTTP server
        metrics_port = config.get("METRICS_PORT", 8000)
        try:
            start_http_server(metrics_port)
            self.logger.info(
                "Prometheus metrics server started",
                extra={"metrics_port": metrics_port}
            )
        except OSError as e:
            self.logger.warning(
                "Metrics server already running or port in use",
                extra={"port": metrics_port, "error": str(e)}
            )
        
        # Initialize Prometheus metrics
        step_name = self.__class__.__name__
        
        # Counters (monotonically increasing)
        self.messages_consumed = Counter(
            'apf_messages_consumed_total',
            'Total number of messages consumed from input',
            ['step', 'survey']
        )
        
        self.messages_processed = Counter(
            'apf_messages_processed_total',
            'Total number of messages successfully processed',
            ['step', 'survey']
        )
        
        self.messages_failed = Counter(
            'apf_messages_failed_total',
            'Total number of messages that failed processing',
            ['step', 'survey', 'error_type']
        )
        
        self.messages_discarded = Counter(
            'apf_messages_discarded_total',
            'Total number of messages discarded',
            ['step', 'survey', 'reason']
        )
        
        # Histograms (distributions with buckets)
        self.processing_time = Histogram(
            'apf_processing_time_seconds',
            'Time spent processing a batch of messages',
            ['step', 'survey'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        self.batch_size = Histogram(
            'apf_batch_size_messages',
            'Number of messages in each batch',
            ['step', 'survey'],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000)
        )
        
        # Gauge (current value, can go up or down)
        self.active_processing = Gauge(
            'apf_active_processing_batches',
            'Number of batches currently being processed',
            ['step', 'survey']
        )
        
        self.logger.info(
            "Metrics initialized",
            extra={"step": step_name, "survey": self.survey}
        )

    def execute(self, messages: list) -> list:
        """
        Process messages with comprehensive logging and metrics.
        """
        start_time = time.time()
        step_name = self.__class__.__name__
        
        # Record batch size metric
        batch_size = len(messages)
        self.batch_size.labels(step=step_name, survey=self.survey).observe(batch_size)
        
        # Mark that we're actively processing
        self.active_processing.labels(step=step_name, survey=self.survey).inc()
        
        # Log batch start
        self.logger.info(
            "Processing batch started",
            extra={"n_messages": batch_size}
        )
        
        processed = []
        for i, msg in enumerate(messages):
            # Count consumed message
            self.messages_consumed.labels(step=step_name, survey=self.survey).inc()
            
            try:
                # Simulate validation
                if not msg.get("id"):
                    # Discard message due to missing required field
                    self.messages_discarded.labels(
                        step=step_name,
                        survey=self.survey,
                        reason="missing_id"
                    ).inc()
                    
                    self.logger.warning(
                        "Message discarded: missing required field",
                        extra={"message_index": i, "field": "id"}
                    )
                    continue
                
                # Simulate processing
                self.logger.debug(
                    "Processing message",
                    extra={"message_index": i, "id": msg.get("id")}
                )
                
                # Example: check for optional field
                if not msg.get("value"):
                    self.logger.warning(
                        "Message missing optional field",
                        extra={"id": msg.get("id"), "field": "value"}
                    )
                
                processed.append(msg)
                
                # Count successfully processed message
                self.messages_processed.labels(step=step_name, survey=self.survey).inc()
                
            except ValueError as e:
                # Count failed message by error type
                self.messages_failed.labels(
                    step=step_name,
                    survey=self.survey,
                    error_type="ValueError"
                ).inc()
                
                self.logger.error(
                    "ValueError while processing message",
                    extra={"message_index": i, "id": msg.get("id"), "error": str(e)},
                    exc_info=True
                )
                
            except Exception as e:
                # Count failed message by generic error
                self.messages_failed.labels(
                    step=step_name,
                    survey=self.survey,
                    error_type="UnknownError"
                ).inc()
                
                self.logger.error(
                    "Unexpected error while processing message",
                    extra={"message_index": i, "id": msg.get("id"), "error": str(e)},
                    exc_info=True
                )
        
        # Record processing time
        duration = time.time() - start_time
        self.processing_time.labels(step=step_name, survey=self.survey).observe(duration)
        
        # Mark that we're done processing
        self.active_processing.labels(step=step_name, survey=self.survey).dec()
        
        # Log batch completion
        self.logger.info(
            "Batch processing complete",
            extra={
                "processed": len(processed),
                "total": batch_size,
                "processing_time_ms": int(duration * 1000)
            }
        )
        
        return processed

    def pre_consume(self):
        """Pre-consumption hook - can be used for initialization"""
        self.logger.info("Step ready to consume messages")
    
    def tear_down(self):
        """Keep metrics endpoint alive for Prometheus to scrape"""
        self.logger.info("Processing complete. Keeping metrics server alive for 120 seconds...")
        time.sleep(120)
        self.logger.info("Shutting down")


if __name__ == "__main__":
    step_config = {
        "LOGGING_LEVEL": "INFO",
        "SURVEY": "LSST",
        "METRICS_PORT": 8000,
        "CONSUMER_CONFIG": {
            "CLASS": "apf.consumers.csv.CSVConsumer",
            "FILE_PATH": "dummy_data.csv",
        },
    }

    step = DummyStepWithMetrics(
        config=step_config,
        level=step_config["LOGGING_LEVEL"]
    )
    
    print("\n" + "="*60)
    print("Dummy Step with Observability Running")
    print("="*60)
    print(f"Logs: stdout + dummy_step.log (JSON format)")
    print(f"Metrics: http://localhost:{step_config['METRICS_PORT']}/metrics")
    print("="*60 + "\n")
    
    step.start()
