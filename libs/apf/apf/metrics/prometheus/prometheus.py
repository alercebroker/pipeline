from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)


class PrometheusMetrics:
    messages_consumed: Counter
    messages_processed: Counter
    batches_processed: Counter
    exceptions: Counter
    last_batch_processed: Gauge
    last_execution_time: Gauge
    batch_size_messages: Histogram
    batch_processing: Histogram

    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self.messages_consumed = Counter(
            "messages_consumed",
            "Number of messages consumed",
            registry=registry,
        )
        self.messages_processed = Counter(
            "messages_processed",
            "Number of messages processed",
            registry=registry,
        )
        self.batches_processed = Counter(
            "batches_processed",
            "Number of batches processed",
            registry=registry,
        )
        self.exceptions = Counter(
            "exceptions",
            "Number of exceptions raised",
            labelnames=["method"],
            registry=registry,
        )
        self.last_batch_processed = Gauge(
            "last_batch_processed",
            "Timestamp of when the last batch finished processing",
            unit="seconds",
            registry=registry,
        )
        self.last_execution_time = Gauge(
            "last_execution_time",
            "Time it took to proccess the most recent batch",
            unit="seconds",
            registry=registry,
        )
        self.batch_size_messages = Histogram(
            "batch_size_messages",
            "Number of messages processed per batch",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000],
            registry=registry,
        )
        self.batch_processing = Histogram(
            "batch_processing",
            "Time taken to process a batch",
            unit="seconds",
            registry=registry,
        )

        exception_labels = [
            "pre_execute",
            "execute",
            "post_execute",
            "pre_produce",
            "produce",
            "post_produce",
            "flush",
        ]

        for exception_label in exception_labels:
            self.exceptions.labels(exception_label).inc(0)
