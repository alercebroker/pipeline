import time
from typing import Any

from apf.core.step import GenericStep

from .client import ImageServiceClient
from .metrics import ServiceMetrics
from .positions import SURVEYS, alert_id, alert_position, clean_position

DEFAULT_MAX_POSITIONS_PER_REQUEST = 200


class HipsDummyStep(GenericStep):
    """Ask the image service for reference images of every alert's position.

    A throughput test: the images are loaded and dropped. Nothing is
    produced or written. The consumer group's lag shows whether the step and
    the service keep up with the stream.
    """

    def __init__(
        self,
        config: dict[str, Any],
        client: ImageServiceClient | None = None,
        service_metrics: ServiceMetrics | None = None,
        **kwargs: Any,
    ):
        super().__init__(config=config, **kwargs)
        self.survey = str(config["SURVEY"]).lower()
        if self.survey not in SURVEYS:
            raise ValueError(
                f"SURVEY must be one of {SURVEYS}, got {config['SURVEY']!r}"
            )
        service_config = config["IMAGE_SERVICE_CONFIG"]
        self.service_metrics = service_metrics or ServiceMetrics()
        self.client = client or ImageServiceClient.from_config(
            service_config, metrics=self.service_metrics
        )
        self.max_positions_per_request = int(
            service_config.get(
                "MAX_POSITIONS_PER_REQUEST", DEFAULT_MAX_POSITIONS_PER_REQUEST
            )
        )
        if self.max_positions_per_request < 1:
            raise ValueError("MAX_POSITIONS_PER_REQUEST must be at least 1")
        self.logger.info(
            f"Survey {self.survey}, bands {self.client.params['bands']}, "
            f"headers {self.client.params['headers']}, "
            f"up to {self.max_positions_per_request} positions per request"
        )

    def pre_execute(
        self, messages: list[dict[str, Any]]
    ) -> list[tuple[float, float]]:
        positions = []
        for message in messages:
            position = clean_position(*alert_position(message, self.survey))
            if position is None:
                self.logger.warning(
                    f"Skipping alert {alert_id(message, self.survey)}: "
                    f"invalid position {alert_position(message, self.survey)}"
                )
                self.service_metrics.positions_skipped.inc()
                continue
            positions.append(position)
        return positions

    def execute(self, positions: list[tuple[float, float]]) -> list:
        started = time.perf_counter()
        n_failed = n_bytes = n_retries = 0
        size = self.max_positions_per_request
        for start in range(0, len(positions), size):
            result = self.client.fetch(positions[start : start + size])
            n_failed += result.n_failed
            n_bytes += result.n_bytes
            n_retries += result.n_retries
        elapsed = time.perf_counter() - started
        self.logger.info(
            f"Got images for {len(positions)} positions "
            f"({n_failed} without images, {n_retries} retries, "
            f"{n_bytes / 1e6:.1f} MB) in {elapsed:.2f} s"
        )
        return []

