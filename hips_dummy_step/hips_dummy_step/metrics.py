"""Prometheus metrics for the calls to the image service."""

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Histogram


class ServiceMetrics:
    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self.request_duration = Histogram(
            "image_service_request_duration",
            "Time from sending a request to having its images loaded, "
            "for requests that succeeded (retries not included)",
            unit="seconds",
            buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120],
            registry=registry,
        )
        self.positions_requested = Counter(
            "image_service_positions_requested",
            "Positions sent to the image service and answered",
            registry=registry,
        )
        self.positions_failed = Counter(
            "image_service_positions_failed",
            "Positions the image service answered without images",
            registry=registry,
        )
        self.positions_skipped = Counter(
            "image_service_positions_skipped",
            "Alerts not sent because their position was missing or invalid",
            registry=registry,
        )
        self.response_bytes = Counter(
            "image_service_response_bytes",
            "Size of successful responses, after decompression",
            registry=registry,
        )
        self.retries = Counter(
            "image_service_retries",
            "Requests retried, by reason",
            labelnames=["reason"],
            registry=registry,
        )
        for reason in ("busy", "server_error", "connection"):
            self.retries.labels(reason).inc(0)
