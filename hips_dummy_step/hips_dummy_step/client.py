"""Client for the image service.

The service takes a list of sky positions and returns, for each one, catalog
reference images of that region of the sky. Request: POST a JSON list of
{"ra": .., "dec": ..} (degrees). Response: an `.npz` with `stamps` (one entry
per position, in order) and `errors` (positions it couldn't serve, still
answered with HTTP 200).

When the service is busy it answers HTTP 429 with a Retry-After header. The
client waits and retries rather than dropping the alerts. Server errors and
connection problems are retried too, with exponential backoff. Any other HTTP
error means the request itself is wrong, so it is raised.
"""

import io
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import requests

from .metrics import ServiceMetrics

logger = logging.getLogger("alerce.ImageServiceClient")

URL_ENV_VAR = "IMAGE_SERVICE_URL"
DEFAULT_RETRY_AFTER = 2.0


@dataclass
class FetchResult:
    n_positions: int
    n_failed: int
    n_bytes: int
    n_retries: int


class ImageServiceClient:
    def __init__(
        self,
        url: str,
        bands: Sequence[str] = ("r",),
        headers: bool = False,
        reference_survey: str | None = None,
        timeout: float = 120.0,
        max_backoff: float = 60.0,
        metrics: ServiceMetrics | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not url:
            raise ValueError("The image service URL is empty")
        if not bands:
            raise ValueError("At least one band is required")
        self.url = url
        self.params = {
            "bands": ",".join(bands),
            "headers": "true" if headers else "false",
        }
        if reference_survey:
            self.params["survey"] = reference_survey
        self.timeout = timeout
        self.max_backoff = max_backoff
        self.metrics = metrics
        self.session = session or requests.Session()
        self.sleep = sleep

    @classmethod
    def from_config(
        cls, config: dict[str, Any], **kwargs: Any
    ) -> "ImageServiceClient":
        """Build a client from the step's IMAGE_SERVICE_CONFIG.

        The URL comes from `URL`, or from the IMAGE_SERVICE_URL environment
        variable when `URL` is not set.
        """
        url = config.get("URL") or os.environ.get(URL_ENV_VAR)
        if not url:
            raise ValueError(
                f"Set IMAGE_SERVICE_CONFIG.URL or the {URL_ENV_VAR} "
                "environment variable"
            )
        return cls(
            url=url,
            bands=config.get("BANDS", ["r"]),
            headers=bool(config.get("HEADERS", False)),
            reference_survey=config.get("REFERENCE_SURVEY"),
            timeout=float(config.get("TIMEOUT", 120.0)),
            max_backoff=float(config.get("MAX_BACKOFF", 60.0)),
            **kwargs,
        )

    def fetch(self, positions: Sequence[tuple[float, float]]) -> FetchResult:
        """Request images for `positions` and load them, then drop them.

        Positions must already be valid (see positions.clean_position).
        """
        body = [{"ra": ra, "dec": dec} for ra, dec in positions]
        response, n_retries, request_seconds = self._post_until_answered(body)

        started = time.perf_counter()
        with np.load(io.BytesIO(response.content), allow_pickle=False) as data:
            stamps = data["stamps"]
            errors = data["errors"]
        if stamps.shape[0] != len(positions):
            raise RuntimeError(
                f"Asked for {len(positions)} positions, "
                f"got images for {stamps.shape[0]}"
            )
        elapsed = request_seconds + (time.perf_counter() - started)

        result = FetchResult(
            n_positions=len(positions),
            n_failed=len(errors),
            n_bytes=len(response.content),
            n_retries=n_retries,
        )
        if self.metrics:
            self.metrics.request_duration.observe(elapsed)
            self.metrics.positions_requested.inc(result.n_positions)
            self.metrics.positions_failed.inc(result.n_failed)
            self.metrics.response_bytes.inc(result.n_bytes)
        return result

    def _post_until_answered(
        self, body: list[dict[str, float]]
    ) -> tuple[requests.Response, int, float]:
        """POST until the service answers 200.

        Return the response, the number of retries, and how long the
        successful attempt took (body download included).
        """
        retries = 0
        backoff_step = 0
        while True:
            started = time.perf_counter()
            try:
                response = self.session.post(
                    self.url,
                    params=self.params,
                    json=body,
                    timeout=self.timeout,
                )
            except (requests.ConnectionError, requests.Timeout) as error:
                wait = self._backoff(backoff_step)
                backoff_step += 1
                self._retrying("connection", wait, repr(error))
                retries += 1
                continue

            if response.status_code == 200:
                return response, retries, time.perf_counter() - started

            if response.status_code == 429:
                wait = self._retry_after(response)
                self._retrying("busy", wait, "service busy (HTTP 429)")
                retries += 1
                continue

            if response.status_code >= 500:
                wait = self._backoff(backoff_step)
                backoff_step += 1
                self._retrying(
                    "server_error",
                    wait,
                    f"HTTP {response.status_code}: {response.text[:200]}",
                )
                retries += 1
                continue

            raise RuntimeError(
                f"The image service rejected the request with HTTP "
                f"{response.status_code}: {response.text[:500]}"
            )

    def _retrying(self, reason: str, wait: float, detail: str):
        if self.metrics:
            self.metrics.retries.labels(reason).inc()
        logger.warning(f"Retrying in {wait:.1f} s: {detail}")
        self.sleep(wait)

    def _backoff(self, step: int) -> float:
        return min(self.max_backoff, 2.0**step)

    @staticmethod
    def _retry_after(response: requests.Response) -> float:
        try:
            return max(0.0, float(response.headers["Retry-After"]))
        except (KeyError, ValueError):
            return DEFAULT_RETRY_AFTER
