"""Persist offline BHRF probabilities into <schema>.probability.

Mirrors feature_writer.py (writing lives here, not in db.py). Turns the BHRF
OutputDTO (flat + hierarchical frames) into DB-ready rows across all 5 seeded
classifiers (ids 5-9). class_name -> class_id uses a {classifier_id: {class_name:
class_id}} map READ FROM THE DB taxonomy table (the authority for the
probability.class_id FK; see db.fetch_taxonomy_maps), passed in so this stays a
pure function. Upsert is ON CONFLICT (oid, sid, classifier_id, class_id) DO UPDATE.
Dry-run unless execute=True.
"""
import logging
from typing import Optional

from sqlalchemy import text

from features.offline import db
from features.offline.classifier_taxonomy_lut import CLASSIFIER_VERSION

log = logging.getLogger(__name__)

# The 5 seeded BHRF classifiers, in id order (flat, top, transient, stochastic, periodic).
CLASSIFIER_IDS = [5, 6, 7, 8, 9]


def classifier_version_to_smallint(version: str) -> int:
    """'2.1.0' -> 210 (production rule). Strips a '_suffix' on the patch part."""
    parts = version.split(".")
    if len(parts) == 3:
        parts[-1] = parts[-1].split("_")[0]
        return int("".join(parts))
    return 0


def _iter_frames(output_dto):
    """Yield (classifier_id, frame) for the 5 BHRF classifiers, in id order."""
    hierarchical = output_dto.hierarchical or {}
    children = hierarchical.get("children", {}) or {}
    return [
        (5, output_dto.probabilities),
        (6, hierarchical.get("top")),
        (7, children.get("Transient")),
        (8, children.get("Stochastic")),
        (9, children.get("Periodic")),
    ]


def build_probability_rows(output_dto, oid: int, lastmjd: float,
                           taxonomy_by_classifier: dict, *,
                           version: str = CLASSIFIER_VERSION, sid: int = 0) -> list:
    """BHRF OutputDTO (single oid) -> DB-ready probability row dicts (all 5 classifiers).

    taxonomy_by_classifier: {classifier_id: {class_name: class_id}} from the DB
    (db.fetch_taxonomy_maps). A class name not in its classifier's map, or a
    classifier with no map at all, raises (mirrors the -1 miss that would store
    garbage). Returns [] for an empty/can't-predict OutputDTO. ranking =
    per-classifier dense rank descending.
    """
    if output_dto is None or output_dto.probabilities is None or len(output_dto.probabilities) == 0:
        return []
    if lastmjd is None:
        raise ValueError("lastmjd is required (probability.lastmjd is NOT NULL)")

    version_smallint = classifier_version_to_smallint(version)
    rows = []
    for classifier_id, frame in _iter_frames(output_dto):
        if frame is None or len(frame) == 0:
            continue
        class_id_of = taxonomy_by_classifier.get(classifier_id)
        if not class_id_of:
            raise ValueError(
                f"no taxonomy map for classifier_id={classifier_id} "
                "(fetch_taxonomy_maps returned nothing for it — is the taxonomy seeded?)"
            )
        series = frame.iloc[0]  # single oid -> Series: class_name -> probability
        ranks = series.rank(ascending=False, method="dense").astype(int)
        for class_name in series.index:
            if class_name not in class_id_of:
                raise ValueError(
                    f"class '{class_name}' not in taxonomy for classifier_id={classifier_id} "
                    "(model/taxonomy mismatch — cannot map to class_id)"
                )
            rows.append({
                "oid": int(oid),
                "sid": int(sid),
                "classifier_id": int(classifier_id),
                "classifier_version": int(version_smallint),
                "class_id": int(class_id_of[class_name]),
                "probability": float(series[class_name]),
                "ranking": int(ranks[class_name]),
                "lastmjd": float(lastmjd),
            })
    return rows
