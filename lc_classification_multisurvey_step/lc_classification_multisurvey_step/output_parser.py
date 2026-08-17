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
        """One message per oid in `model_output.probabilities`.

        Caller contract, as for `build_probability_rows`: each head's frame must
        have a unique oid index. Duplicate oids are not collapsed here;
        de-duplication happens upstream when the features frame is built.

        A head is absent from an oid's `top_class` (never present as null) when
        the head is missing, has no classes, does not cover that oid, or scored
        it entirely NaN. Per design §8 these drop the affected (oid, head) entry
        instead of killing the batch.
        """
        if model_output is None or model_output.probabilities is None:
            return KafkaOutput([])
        if len(model_output.probabilities) == 0:
            return KafkaOutput([])

        # Rank each head once for the whole frame. The per-oid form (a .loc plus
        # an idxmax per oid per head) costs ~24x more on a 1000-object batch
        # (344 ms vs 15 ms, same output).
        heads = []
        for name, frame in iter_head_frames(model_output, base_name):
            # shape[1], not len(): a frame with rows but no columns has no
            # argmax and would raise "argmax of an empty sequence".
            if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                continue
            # An all-NaN row has no argmax either: pandas 2 returns NaN (an
            # opaque KeyError downstream), pandas 3 raises outright. Dropping
            # those rows leaves the oid simply uncovered by this head.
            frame = frame.dropna(how="all")
            if frame.shape[0] == 0:
                continue
            # Plain dicts, not Series: the per-oid lookup below is then a hash
            # rather than a pandas label lookup, which is ~4x cheaper again.
            heads.append(
                (name, frame.idxmax(axis=1).to_dict(), frame.max(axis=1).to_dict())
            )

        messages = []
        for oid in model_output.probabilities.index:
            top_class = {}
            for name, class_names, probabilities in heads:
                if oid not in class_names:
                    continue
                top_class[name] = {
                    "class_name": str(class_names[oid]),
                    "probability": float(probabilities[oid]),
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
