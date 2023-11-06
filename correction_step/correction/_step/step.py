from __future__ import annotations
import pickle
import json
import logging

import pandas as pd
from apf.core import get_class
from apf.core.step import GenericStep

from ..core.corrector import Corrector

from pprint import pprint

class CorrectionStep(GenericStep):
    """Step that applies magnitude correction to new alert and previous candidates.

    The correction refers to passing from the magnitude measured from the flux in the difference
    stamp to the actual apparent magnitude. This requires a reference magnitude to work.
    """

    def __init__(
        self,
        config,
        **step_args,
    ):
        super().__init__(config=config, **step_args)
        cls = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(self.config["SCRIBE_PRODUCER_CONFIG"])
        self.set_producer_key_field("aid")

    @staticmethod
    def create_step() -> CorrectionStep:
        import os
        from .settings import settings_creator
        from prometheus_client import start_http_server
        from apf.metrics.prometheus import PrometheusMetrics, DefaultPrometheusMetrics

        settings = settings_creator()
        level = logging.INFO
        if os.getenv("LOGGING_DEBUG"):
            level = logging.DEBUG

        logger = logging.getLogger("alerce")
        logger.setLevel(level)

        fmt = logging.Formatter("%(asctime)s %(levelname)7s %(name)36s: %(message)s", "%Y-%m-%d %H:%M:%S")
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        handler.setLevel(level)

        logger.addHandler(handler)

        prometheus_metrics = PrometheusMetrics() if settings["PROMETHEUS"] else DefaultPrometheusMetrics()
        if settings["PROMETHEUS"]:
            start_http_server(8000)

        return CorrectionStep(config=settings, prometheus_metrics=prometheus_metrics)

    @classmethod
    def pre_produce(cls, result: dict):
        detections = pd.DataFrame(result["detections"]).groupby("aid")
        try:  # At least one non-detection
            non_detections = pd.DataFrame(result["non_detections"]).groupby("aid")
        except KeyError:  # to reproduce expected error for missing non-detections in loop
            non_detections = pd.DataFrame(columns=["aid"]).groupby("aid")
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
        return {
            "detections": detections,
            "non_detections": non_detections.to_dict("records"),
            "coords": coords,
        }

    def post_execute(self, result: dict):
        self.produce_scribe(result["detections"])
        return result

    def produce_scribe(self, detections: list[dict]):
        for detection in detections:
            detection = detection.copy()  # Prevent further modification for next step
            if not detection.pop("new"):
                continue
            candid = detection.pop("candid")
            is_forced = detection.pop("forced")
            set_on_insert = not detection.get("has_stamp", False)
            extra_fields = detection["extra_fields"].copy()
            # remove possible elasticc extrafields
            for to_remove in ["prvDiaSources", "prvDiaForcedSources", "fp_hists"]:
                if to_remove in extra_fields:
                    extra_fields.pop(to_remove)

            if "diaObject" in extra_fields:
                extra_fields["diaObject"] = pickle.loads(extra_fields["diaObject"])

            detection["extra_fields"] = extra_fields
            scribe_data = {
                "collection": "forced_photometry" if is_forced else "detection",
                "type": "update",
                "criteria": {"_id": candid, "candid": candid},
                "data": detection,
                "options": {"upsert": True, "set_on_insert": set_on_insert},
            }
            scribe_payload = {"payload": json.dumps(scribe_data)}
            self.scribe_producer.produce(scribe_payload)
