"""Assembly layer: turn one correction-ztf message into a features frame.

Implements the pure message->features logic of the production feature_step
(`feature_step/features/step.py`) without its Kafka/scribe plumbing. Uses
the real pipeline parser and lc_classifier modules directly (no vendoring).
"""
from features.utils.parsers import detections_to_astro_object
from lc_classifier.features.core.base import discard_bogus_detections
from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor


def _prepare_detections(message: dict, min_detections: int = 1):
    """pre_execute + execute packing: drop bogus, enforce min real detections,
    add `index_column` and `aid=oid` (the message has no aid). Returns the
    detection dicts for the parser, or None if too few real detections.

    discard_bogus_detections tolerates int procstatus (fixed on this branch).
    """
    dets = discard_bogus_detections(message.get("detections", []))
    n_real = sum(1 for d in dets if not d.get("forced", False))
    if n_real < min_detections:
        return None
    return [
        {**d, "aid": d["oid"], "index_column": f'{d["measurement_id"]}_{d["oid"]}'}
        for d in dets
    ]


def _xmatches(allwise):
    """Build the AllWISE xmatches dict the parser reads. Returns None when no
    AllWISE row is available. Stage-6: validate that leaving AllWISE absent
    (returning None) matches the live pipeline's W1-W4 NaN behavior."""
    if allwise is None or len(allwise) == 0:
        return None
    row = allwise.iloc[0]
    return {
        "allwise": {},  # presence gates the parser's W1-W4 read
        "metadata": {
            "w1mpro": {"Float64": row["W1"]},
            "w2mpro": {"Float64": row["W2"]},
            "w3mpro": {"Float64": row["W3"]},
            "w4mpro": {"Float64": row["W4"]},
        },
    }


def message_to_astro_object(message: dict, references_db, allwise, min_detections: int = 1):
    """Correction-ztf message -> AstroObject via the real pipeline parser. All epochs
    enter through the `detections` arg with `forced=[]`; the per-row `forced` flag
    routes forced epochs to `forced_photometry`. Returns None if the message has too
    few real detections."""
    dets = _prepare_detections(message, min_detections)
    if dets is None:
        return None
    return detections_to_astro_object(dets, [], _xmatches(allwise), references_db)


def compute_astro_object(message: dict, references_db, allwise, min_detections: int = 1,
                         preprocessor=None, extractor=None):
    """Per-oid path: message -> AstroObject -> preprocess -> extract -> AstroObject.

    `preprocessor`/`extractor` are injectable for tests; defaults are the production
    stack, constructed lazily (the extractor is heavy). Returns the post-extract
    AstroObject, or None if the message has too few real detections."""
    if preprocessor is None:
        preprocessor = ZTFLightcurvePreprocessor(drop_bogus=True)
    if extractor is None:
        extractor = ZTFFeatureExtractor()

    ao = message_to_astro_object(message, references_db, allwise, min_detections)
    if ao is None:
        return None
    preprocessor.preprocess_single_object(ao)
    extractor.compute_features_single_object(ao)
    return ao


def compute_features(message: dict, references_db, allwise, min_detections: int = 1,
                     preprocessor=None, extractor=None):
    """Per-oid path: message -> AstroObject -> preprocess -> extract -> features.

    Returns the long features frame, or None if the message has too few real
    detections."""
    ao = compute_astro_object(message, references_db, allwise, min_detections,
                              preprocessor=preprocessor, extractor=extractor)
    if ao is None:
        return None
    return ao.features
