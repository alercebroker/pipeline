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
from .input_dto import create_input_dto, filter_messages, lastmjd_by_oid
from .output_parser import MultisurveyOutputParser
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

        # Startup, in order: names -> ids, then ids -> taxonomy. Both are
        # read-only and cached; the connection is not used again. Any of the four
        # §8 assertions failing raises here and the step refuses to start.
        #
        # NullPool because this connection serves exactly two queries and is then
        # idle for the life of the consumer. With the default QueuePool the
        # startup checkout is returned to the pool rather than closed, holding an
        # idle Postgres connection per replica against max_connections for
        # nothing. correction_multisurvey_step/step.py does the same.
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

        self.step_parser = MultisurveyOutputParser()

    @staticmethod
    def _empty_output() -> OutputDTO:
        return OutputDTO(pd.DataFrame(), {"top": pd.DataFrame(), "children": {}})

    def execute(self, messages: List[dict]) -> Tuple[OutputDTO, dict]:
        kept = filter_messages(messages, self.min_detections)
        self.logger.info(f"Classifying {len(kept)}/{len(messages)} messages")
        if not kept:
            return self._empty_output(), {}

        dto = create_input_dto(kept)
        can_predict, reason = self.model.can_predict(dto)
        if not can_predict:
            self.logger.warning(f"Model cannot predict this batch: {reason}")
            return self._empty_output(), {}

        try:
            output_dto = self.model.predict(dto)
        except Exception as e:
            self.logger.error(f"Prediction failed for this batch: {e}")
            self.logger.error(traceback.format_exc())
            return self._empty_output(), {}

        return output_dto, lastmjd_by_oid(kept)

    def post_execute(self, result: Tuple[OutputDTO, dict]) -> Tuple[OutputDTO, dict]:
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
        """
        if not rows:
            return

        last_index = len(rows) - 1
        for index, row in enumerate(rows):
            command = {"step": "update-probability", "survey": "ztf", "payload": row}
            self.scribe_producer.produce(
                {"payload": json.dumps(command)},
                key=str(row["oid"]).encode("utf-8"),
                on_delivery=None,
            )
            if index == last_index:
                self.scribe_producer.producer.flush()

        self.logger.info(f"Produced {len(rows)} probability rows to the scribe")

    def pre_produce(self, result: Tuple[OutputDTO, dict]):
        # PLACEHOLDER downstream payload — design doc §9.
        return self.step_parser.parse(
            result[0], base_name=self.classifier_name, version=self.model_version
        ).value

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
