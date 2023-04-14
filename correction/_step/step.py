from __future__ import annotations
import json
import logging

import pandas as pd
from apf.core import get_class
from apf.core.step import GenericStep

from ..core.corrector import Corrector


class CorrectionStep(GenericStep):
    """Step that applies magnitude correction to new alert and previous candidates.

    The correction refers to passing from the magnitude measured from the flux in the difference
    stamp to the actual apparent magnitude. This requires a reference magnitude to work.
    """

    def __init__(
        self,
        config,
        level=logging.INFO,
        **step_args,
    ):
        super().__init__(config=config, level=level, **step_args)
        cls = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(self.config["SCRIBE_PRODUCER_CONFIG"])
        self.set_producer_key_field("aid")

    @staticmethod
    def create_step() -> CorrectionStep:
        from .settings import settings_creator
        from prometheus_client import start_http_server
        from apf.metrics.prometheus import PrometheusMetrics, DefaultPrometheusMetrics

        settings = settings_creator()
        level = logging.DEBUG if settings["LOGGING_DEBUG"] else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        prometheus_metrics = PrometheusMetrics() if settings["PROMETHEUS"] else DefaultPrometheusMetrics()
        if settings["PROMETHEUS"]:
            start_http_server(8000)

        return CorrectionStep(config=settings, level=level, prometheus_metrics=prometheus_metrics)

    @classmethod
    def pre_produce(cls, result: dict):
        detections = pd.DataFrame(result["detections"]).groupby("aid")
        non_detections = pd.DataFrame(result["non_detections"]).groupby("aid")
        output = []
        for aid, dets in detections:
            try:
                nd = non_detections.get_group(aid).to_dict("records")
            except KeyError:
                nd = []
            output.append(
                {
                    "aid": aid,
                    "meanra": result["coords"][aid]["meanra"],
                    "meandec": result["coords"][aid]["meandec"],
                    "detections": dets.to_dict("records"),
                    "non_detections": nd,
                }
            )
        return output

    @classmethod
    def pre_execute(cls, messages: list[dict]) -> dict:
        detections, non_detections = [], []
        for msg in messages:
            detections.extend(msg["detections"])
            non_detections.extend(msg["non_detections"])
        return {"detections": detections, "non_detections": non_detections}

    @classmethod
    def execute(cls, message: dict) -> dict:
        corrector = Corrector(message["detections"])
        detections = corrector.corrected_as_records()
        non_detections = pd.DataFrame(message["non_detections"]).drop_duplicates(["oid", "fid", "mjd"])
        coords = corrector.coordinates_as_records()
        return {"detections": detections, "non_detections": non_detections.to_dict("records"), "coords": coords}

    def post_execute(self, result: dict):
        self.produce_scribe(result["detections"])
        return result

    def produce_scribe(self, detections: list[dict]):
        for detection in detections:
            detection = detection.copy()  # Prevent modification for next step
            candid = detection.pop("candid")
            set_on_insert = not detection.get("has_stamp", False)
            scribe_data = {
                "collection": "detection",
                "type": "update",
                "criteria": {"_id": candid},
                "data": detection,
                "options": {"upsert": True, "set_on_insert": set_on_insert},
            }
            scribe_payload = {"payload": json.dumps(scribe_data)}
            self.scribe_producer.produce(scribe_payload)
