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

# Consecutive dropped batches before the log escalates to CRITICAL.
#
# Dropping a bad batch is what design §8 asks for, but every drop still commits
# its offset, so a *persistent* fault — a broken lazy import inside
# `create_input_dto`, say — consumes the topic at full speed, writes nothing,
# and still reports healthy. One poison batch is routine and must not page
# anyone; this many in a row is systemic and needs a human before the backlog
# has to be replayed. At the default `consume.messages` of 100 this is ~500
# objects discarded, which is small enough to catch early and large enough that
# a single bad batch or a brief blip never trips it.
CONSECUTIVE_DROP_ALERT = 5


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

        # Batches dropped back-to-back; see CONSECUTIVE_DROP_ALERT.
        self._consecutive_drops = 0
        # Whether the batch in flight has already been dropped; see `pre_execute`.
        self._batch_dropped = False

    @staticmethod
    def _empty_output() -> OutputDTO:
        return OutputDTO(pd.DataFrame(), {"top": pd.DataFrame(), "children": {}})

    def pre_execute(self, messages: List[dict]) -> List[dict]:
        """Passthrough; the batch marker for the drop accounting is set here.

        The counter has to be reset by a batch that got all the way through, not
        by one that merely got out of `execute`: with the reset sitting at the end
        of `execute`, a fault in `post_execute` or `pre_produce` would be counted
        and then immediately un-counted by the next batch's successful `execute`,
        so the escalation could never fire for the very stages this marker was
        added to cover. `pre_produce` clears the counter instead, and only if this
        flag is still false. apf calls `pre_execute` first for every batch.
        """
        self._batch_dropped = False
        return messages

    def _record_drop(self, what: str, error: Exception) -> None:
        """Log an unexpected failure and count it against the drop budget (§8).

        The caller then yields whatever "nothing" means for its stage, so the step
        produces nothing, commits, and moves on. Letting the exception escape
        instead would only log and re-raise (apf `core/step.py`), exiting the
        process with the offset uncommitted — Kafka then redelivers the same
        batch, which fails identically, and the partition stalls forever on one
        poison message.

        The cost of that trade is that a *persistent* fault is silent: offsets
        keep advancing and the step looks healthy. The drop counter and the
        escalation below exist to make that visible, and are the reason dropping
        is safe. This method must never raise — it runs on the path that already
        failed — so the metric is best-effort.
        """
        self._batch_dropped = True
        self._consecutive_drops += 1
        self.logger.error(f"{what} for this batch, dropping it: {error}")
        self.logger.error(traceback.format_exc())

        try:
            # One label for every drop site, not one per stage: an operator wants
            # a single "this step is discarding batches" series to alert on, and
            # `what` already names the stage in the log line above.
            self.metrics.exceptions.labels("execute_drop").inc()
        except Exception:
            # Telemetry must never be the thing that kills a batch.
            self.logger.warning("Could not record the dropped-batch metric", exc_info=True)

        if self._consecutive_drops >= CONSECUTIVE_DROP_ALERT:
            self.logger.critical(
                f"{self._consecutive_drops} consecutive batches dropped and their "
                "offsets committed: this topic is being consumed and discarded, and "
                "the step will keep reporting healthy. Output is being lost; the "
                "skipped batches need a replay once the cause is fixed."
            )

    def _drop_batch(self, what: str, error: Exception) -> Tuple[OutputDTO, dict]:
        """`_record_drop` plus the empty result the pre-produce stages return."""
        self._record_drop(what, error)
        return self._empty_output(), {}

    def execute(self, messages: List[dict]) -> Tuple[OutputDTO, dict]:
        try:
            kept = filter_messages(messages, self.min_detections)
        except Exception as error:
            return self._drop_batch("Filtering failed", error)

        self.logger.info(f"Classifying {len(kept)}/{len(messages)} messages")
        if not kept:
            # A normal outcome, not a drop: nothing in the batch was classifiable.
            # `_batch_dropped` stays false, so `pre_produce` clears the counter.
            return self._empty_output(), {}

        try:
            dto = create_input_dto(kept)
        except Exception as error:
            return self._drop_batch("Building the input DTO failed", error)

        # Deliberately unguarded: `can_predict` is null checks over the features
        # frame and has no failure mode of its own.
        can_predict, reason = self.model.can_predict(dto)
        if not can_predict:
            self.logger.warning(f"Model cannot predict this batch: {reason}")
            return self._empty_output(), {}

        try:
            output_dto = self.model.predict(dto)
        except Exception as error:
            return self._drop_batch("Prediction failed", error)

        try:
            lastmjd_map = lastmjd_by_oid(kept)
        except Exception as error:
            return self._drop_batch("Computing lastmjd failed", error)

        return output_dto, lastmjd_map

    def post_execute(self, result: Tuple[OutputDTO, dict]) -> Tuple[OutputDTO, dict]:
        """Build the probability rows, then write them to the scribe.

        The two halves are wrapped differently on purpose — see below. apf
        re-raises from `_post_execute` and that call is not inside a `try` in
        `start()`, so anything that escapes here exits the process with the offset
        uncommitted, which is the permanent partition stall `_drop_batch` exists
        to prevent.
        """
        output_dto, lastmjd_map = result

        try:
            rows = build_probability_rows(
                output_dto,
                lastmjd_map,
                self.classifier_ids,
                self.taxonomy_maps,
                base_name=self.classifier_name,
                version=self.model_version,
                sid=self.sid,
            )
        except Exception as error:
            # Guarded, on the same reasoning as `execute`'s guards: this is
            # deterministic pure computation over the batch, so a fault is
            # reproduced exactly on every redelivery and the partition stalls
            # forever. Dropping is the only outcome that makes progress.
            return self._drop_batch("Building the probability rows failed", error)

        # Deliberately NOT guarded, and this asymmetry is the point: everything
        # above is pure computation, `produce_scribe` is I/O. A broker fault is
        # typically transient, so letting it propagate leaves the offset
        # uncommitted and the batch is retried — correct at-least-once behaviour,
        # and exactly what apf's flush guard in `_post_produce` is for. Swallowing
        # it would silently lose scribe writes, which is strictly worse than a
        # stall: the scribe is this step's real output (design §2, decision 3), so
        # a dropped write is data that no replay knows to go back for. Do not
        # "complete" the guard by wrapping this call.
        self.produce_scribe(rows)
        return result

    def produce_scribe(self, rows: List[dict]) -> None:
        """One `update-probability` command per row, keyed by oid.

        Envelope matches stamp_classifier_2025_multisurvey_step and is accepted by
        scribe_multisurvey's `decode.command_factory` (design doc §7).

        No explicit flush here. apf's `_post_produce` drains every producer it
        finds on the step before the consumer offset is committed, and the
        `KafkaProducer.flush` it calls honours `FLUSH_TIMEOUT` and raises
        `BufferError` rather than committing over undelivered writes. Reaching
        past it to `self.scribe_producer.producer.flush()` — as the older sibling
        steps do — bypasses both guards: it ignores `FLUSH_TIMEOUT`, blocks for
        the full `message.timeout.ms` when the broker is down, and reports the
        outage as a `post_execute` error.
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

    def pre_produce(self, result: Tuple[OutputDTO, dict]):
        # PLACEHOLDER downstream payload — design doc §9.
        try:
            payload = self.step_parser.parse(
                result[0], base_name=self.classifier_name, version=self.model_version
            ).value
        except Exception as error:
            # Guarded for the same reason as `build_probability_rows`: pure
            # deterministic computation, so apf re-raising from `_pre_produce`
            # (also outside any `try` in `start()`) would stall the partition on a
            # fault that repeats identically forever.
            #
            # The empty list is what "produce nothing" means at this stage: apf's
            # `produce()` iterates whatever it is given, so `[]` emits no messages
            # and, with no PRODUCER_CONFIG, the DefaultProducer is a no-op anyway.
            #
            # Note the scribe write in `post_execute` has already succeeded by
            # now, so a drop here loses only the §9 placeholder, which nothing
            # consumes. It still counts against the budget: a persistent parse
            # fault is a real bug and the counter is what surfaces it.
            self._record_drop("Parsing the downstream payload failed", error)
            payload = []

        # Last stage the step guards, so this is where a clean batch clears the
        # counter — see `pre_execute` for why it cannot be done in `execute`.
        if not self._batch_dropped:
            self._consecutive_drops = 0
        return payload

    def tear_down(self):
        # No `else: self.consumer.__del__()`. `__del__` and `teardown` are defined
        # only on KafkaConsumer, so the sibling steps' else-branch raises
        # AttributeError on a JSON/AVRO replay consumer, which `_tear_down`
        # re-raises before `_write_success()` runs. Nothing is lost by omitting
        # it: KafkaConsumer.__del__ only calls teardown() anyway.
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
