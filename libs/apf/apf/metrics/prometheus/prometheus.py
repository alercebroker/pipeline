from prometheus_client import REGISTRY, CollectorRegistry, Summary


class PrometheusMetrics:
    consumed_messages: Summary
    processed_messages: Summary
    execution_time: Summary

    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self.consumed_messages = Summary(
            "consumed_messages",
            "Number of messages consumed",
            registry=registry,
        )
        self.processed_messages = Summary(
            "processed_messages",
            "Number of messages processed",
            registry=registry,
        )
        self.execution_time = Summary(
            "execution_time",
            "Execution time of the batch",
            registry=registry,
        )
