"""PLACEHOLDER downstream producer payload.

Decision 5 of the design doc is "new multisurvey output schema", but its shape is
deferred — see §9. This emits a minimal per-oid message (oid, classifier name and
version, and the top-ranked class per head) so the step's produce stage is wired
end to end. It is NOT a contract: no schemas/lc_classification_multisurvey_step/
avsc exists, and nothing downstream should be pointed at this topic until the
schema is designed. Deferring is safe because the scribe is the real output path
(decision 3).

Duck-typed over the OutputDTO like probabilities.py, so it needs no
alerce_classifiers import.
"""
from dataclasses import dataclass
from typing import Generic, TypeVar

from .probabilities import DEFAULT_CLASSIFIER_NAME, iter_head_frames

T = TypeVar("T")


@dataclass
class KafkaOutput(Generic[T]):
    value: T


class MultisurveyOutputParser:
    """OutputDTO -> placeholder downstream messages."""

    def parse(
        self, model_output, base_name: str = DEFAULT_CLASSIFIER_NAME, version: str = "", **kwargs
    ) -> KafkaOutput:
        if model_output is None or model_output.probabilities is None:
            return KafkaOutput([])
        if len(model_output.probabilities) == 0:
            return KafkaOutput([])

        heads = [
            (name, frame)
            for name, frame in iter_head_frames(model_output, base_name)
            if frame is not None and len(frame) > 0
        ]

        messages = []
        for oid in model_output.probabilities.index:
            top_class = {}
            for name, frame in heads:
                if oid not in frame.index:
                    continue
                series = frame.loc[oid]
                class_name = series.idxmax()
                top_class[name] = {
                    "class_name": str(class_name),
                    "probability": float(series[class_name]),
                }
            messages.append(
                {
                    "oid": int(oid),
                    "classifier_name": base_name,
                    "classifier_version": version,
                    "top_class": top_class,
                }
            )
        return KafkaOutput(messages)
