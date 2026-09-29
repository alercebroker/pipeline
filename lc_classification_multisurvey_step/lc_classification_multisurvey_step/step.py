"""Multisurvey LC classification step (ZTF BHRF / Squidward).

Consumes the multisurvey feature_step output topic, classifies the batch, and
produces one `update-probability` command per probability row to the
scribe_multisurvey topic. The step writes nothing to the database — the scribe
owns the upsert (design doc §2, decision 3).
"""
import json
import logging
import traceback
from typing import List, Tuple

import numexpr
import pandas as pd
from apf.consumers import KafkaConsumer
from apf.core import get_class
from apf.core.step import GenericStep
from alerce_classifiers.base.dto import OutputDTO

from .db.db import PSQLConnection, resolve_classifiers
from .input_dto import collapse_by_oid, create_input_dto, filter_messages, lastmjd_by_oid
from .probabilities import build_probability_rows, head_names


class LateClassifierMultisurvey(GenericStep):
    """BHRF classification over the multisurvey feature stream."""

    def __init__(self, config={}, level=logging.INFO, **step_args):
        super().__init__(config=config, level=level, **step_args)
        numexpr.utils.set_num_threads(1)

        model_config = config["MODEL_CONFIG"]
        self.classifier_name = model_config["CLASSIFIER_NAME"]
        self.model_version = model_config["VERSION"]
        self.sid = int(model_config.get("SID", 0))
        self.min_detections = model_config.get("MIN_DETECTIONS")

        # Two read-only startup queries, then the connection is idle for the life
        # of the consumer — hence NullPool, so no idle Postgres connection is held
        # per replica (correction_multisurvey_step does the same).
        self.db = PSQLConnection(config["PSQL_CONFIG"], poolclass="NullPool")
        self.classifier_ids, self.taxonomy_maps = resolve_classifiers(
            head_names(self.classifier_name), self.model_version, self.db
        )

        self.mapper = get_class(model_config["CLASS_MAPPER"])()
        self.model = get_class(model_config["CLASS"])(
            **{"mapper": self.mapper, **model_config["PARAMS"]}
        )

        scribe_config = config["SCRIBE_PRODUCER_CONFIG"]
        self.scribe_producer = get_class(scribe_config["CLASS"])(scribe_config)

    @staticmethod
    def _empty_output() -> OutputDTO:
        return OutputDTO(pd.DataFrame(), {"top": pd.DataFrame(), "children": {}})

    def predict(self, model_input) -> OutputDTO | None:
        try:
            return self.model.predict(model_input)
        except Exception as error:
            self.logger.error(error)
            self.logger.error(traceback.format_exc())

    def pre_execute(self, messages: List[dict]) -> dict:
        """Filter (min_detections gate) and collapse to one message per oid.

        Collapsed once here so the features frame and the lastmjd map cannot
        disagree about which message won for a duplicated oid.
        """
        kept = filter_messages(messages, self.min_detections)
        collapsed = collapse_by_oid(kept)
        self.logger.info(
            f"Classifying {len(collapsed)} objects ({len(kept)}/{len(messages)} messages kept)"
        )
        return collapsed

    def execute(self, collapsed: dict) -> Tuple[OutputDTO, dict]:
        """Classify the batch; returns the model output and the lastmjd per oid."""
        if not collapsed:
            return self._empty_output(), {}

        dto = create_input_dto(collapsed)

        can_predict, reason = self.model.can_predict(dto)
        if not can_predict:
            self.logger.warning(f"Model cannot predict this batch: {reason}")
            return self._empty_output(), {}

        output_dto = self.predict(dto)
        if output_dto is None:
            return self._empty_output(), {}

        return output_dto, lastmjd_by_oid(collapsed)

    def post_execute(self, result: Tuple[OutputDTO, dict]) -> Tuple[OutputDTO, dict]:
        """Build the probability rows and write them to the scribe."""
        output_dto, lastmjd_map = result
        rows = build_probability_rows(
            output_dto,
            lastmjd_map,
            self.classifier_ids,
            self.taxonomy_maps,
            base_name=self.classifier_name,
            version=self.model_version,
            sid=self.sid,
        )
        self.produce_scribe(rows)
        return result

    def produce_scribe(self, rows: List[dict]) -> None:
        """One `update-probability` command per row, keyed by oid.

        Envelope matches stamp_classifier_2025_multisurvey_step and is accepted by
        scribe_multisurvey's `decode.command_factory` (design doc §7).

        No explicit flush: apf's `_post_produce` drains every producer before the
        offset is committed, honouring FLUSH_TIMEOUT. Do not reach past it to
        `self.scribe_producer.producer.flush()` as the sibling steps do — that
        blocks for the full `message.timeout.ms` when the broker is down.
        """
        if not rows:
            return

        for row in rows:
            command = {"step": "update-probability", "survey": "ztf", "payload": row}
            self.scribe_producer.produce(
                {"payload": json.dumps(command)},
                key=str(row["oid"]).encode("utf-8"),
            )

        self.logger.info(f"Produced {len(rows)} probability rows to the scribe")

    def pre_produce(self, result: Tuple[OutputDTO, dict]) -> list:
        # No downstream output yet (design doc §9): PRODUCER_CONFIG is always {},
        # and returning the raw result would have apf iterate the tuple as if it
        # were messages. The scribe is the only output path.
        return []

    def tear_down(self):
        # No `else: self.consumer.__del__()`: that branch raises AttributeError on
        # a JSON/AVRO replay consumer, and KafkaConsumer.__del__ only calls
        # teardown() anyway.
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
