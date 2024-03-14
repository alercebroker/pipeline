from __future__ import annotations
import pickle
import json
import logging
from copy import deepcopy

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
        **step_args,
    ):
        super().__init__(config=config, **step_args)
        cls = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(self.config["SCRIBE_PRODUCER_CONFIG"])
        self.set_producer_key_field("oid")

    @staticmethod
    def create_step() -> CorrectionStep:
        import os
        from prometheus_client import start_http_server
        from apf.metrics.prometheus import PrometheusMetrics, DefaultPrometheusMetrics
        from apf.core.settings import config_from_yaml_file

        if os.getenv("CONFIG_FROM_YAML"):
            settings = config_from_yaml_file("/config/config.yaml")
        else:
            from .settings import settings_creator

            settings = settings_creator()
        level = logging.INFO
        if settings.get("LOGGING_DEBUG"):
            level = logging.DEBUG

        logger = logging.getLogger("alerce")
        logger.setLevel(level)

        fmt = logging.Formatter(
            "%(asctime)s %(levelname)7s %(name)36s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        handler.setLevel(level)

        logger.addHandler(handler)

        if settings["FEATURE_FLAGS"]["USE_PROFILING"]:
            from pyroscope import configure

            configure(
                application_name="step.Correction",
                server_address=settings["PYROSCOPE_SERVER"],
            )

        prometheus_metrics = (
            PrometheusMetrics()
            if settings["FEATURE_FLAGS"]["PROMETHEUS"]
            else DefaultPrometheusMetrics()
        )
        if settings["FEATURE_FLAGS"]["PROMETHEUS"]:
            start_http_server(8000)

        return CorrectionStep(config=settings, prometheus_metrics=prometheus_metrics)

    @classmethod
    def pre_produce(cls, result: dict):
        result["detections"] = pd.DataFrame(result["detections"]).groupby("oid")
        try:  # At least one non-detection
            result["non_detections"] = pd.DataFrame(result["non_detections"]).groupby("oid")
        except KeyError:  # to reproduce expected error for missing non-detections in loop
            result["non_detections"] = pd.DataFrame(columns=["oid"]).groupby("oid")
        output = []
        for oid, dets in result["detections"]:
            output_message = {
                "oid": oid,
                "candid": result["candids"][oid],
                "meanra": result["coords"][oid]["meanra"],
                "meandec": result["coords"][oid]["meandec"],
                "detections": dets.to_dict("records"),
            }
            try:
                output_message["non_detections"] = (
                    result["non_detections"].get_group(oid).to_dict("records")
                )
            except KeyError:
                output_message["non_detections"] = []
            output.append(output_message)
        return output

    @classmethod
    def pre_execute(cls, messages: list[dict]) -> dict:
        detections, non_detections = [], []
        candids = {}
        for msg in messages:
            if msg["oid"] not in candids:
                candids[msg["oid"]] = []
            candids[msg["oid"]].extend(msg["candid"])
            detections.extend(msg["detections"])
            non_detections.extend(msg["non_detections"])
        return {"detections": detections, "non_detections": non_detections, "candids": candids}

    @classmethod
    def execute(cls, message: dict) -> dict:
        corrector = Corrector(message["detections"])
        detections = corrector.corrected_as_records()
        coords = corrector.coordinates_as_records()
        non_detections = pd.DataFrame(message["non_detections"]).drop_duplicates(
            ["oid", "fid", "mjd"]
        )
        return {
            "detections": detections,
            "non_detections": non_detections.to_dict("records"),
            "coords": coords,
            "candids": message["candids"],
        }

    def post_execute(self, result: dict):
        self.produce_scribe(result["detections"])
        return result

    def produce_scribe(self, detections: list[dict]):
        count = 0
        for detection in detections:
            count += 1
            flush = False
            # Prevent further modification for next step
            detection = deepcopy(detection)
            if not detection.pop("new"):
                continue
            candid = detection.pop("candid")
            oid = detection.get("oid")
            is_forced = detection.pop("forced")
            set_on_insert = not detection.get("has_stamp", False)
            extra_fields = detection["extra_fields"]
            # remove possible elasticc extrafields
            for to_remove in ["prvDiaSources", "prvDiaForcedSources", "fp_hists"]:
                extra_fields.pop(to_remove, None)
            if "diaObject" in extra_fields:
                extra_fields["diaObject"] = pickle.loads(extra_fields["diaObject"])
            detection["extra_fields"] = extra_fields
            scribe_data = {
                "collection": "forced_photometry" if is_forced else "detection",
                "type": "update",
                "criteria": {"candid": candid, "oid": oid},
                "data": detection,
                "options": {"upsert": True, "set_on_insert": set_on_insert},
            }
            scribe_payload = {"payload": json.dumps(scribe_data)}
            if count == len(detections):
                flush = True
            self.scribe_producer.produce(scribe_payload, flush=flush, key=oid)
