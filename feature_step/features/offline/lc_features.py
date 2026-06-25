"""Assembly layer: turn one magstats_ms_step ZTF output message into a features frame.

Implements the pure message->features logic of the production feature_step
(`feature_step/features/step.py`) without its Kafka/scribe plumbing. Uses
the real pipeline parser and lc_classifier modules directly (no vendoring).
"""
from importlib.metadata import version as _pkg_version

from features.utils.parsers import detections_to_astro_object, prepare_ao_features_for_db
from features.offline.feature_lut import load_feature_name_lut, version_name_to_id
from lc_classifier.features.core.base import discard_bogus_detections
from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor

SID_ZTF = 0


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
    """magstats_ms_step ZTF output message -> AstroObject via the real pipeline parser. All epochs
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


def compute_db_features(message: dict, references_db, allwise, min_detections: int = 1,
                        preprocessor=None, extractor=None, feature_name_lut=None,
                        version_name=None):
    """Per-oid path -> DB-ready feature rows, following the production save rules.

    Returns a DataFrame with exactly the `feature` table columns
    [oid, sid, feature_id, band, version, value] (NaN values dropped,
    fid->band code, name->feature_id via the LUT), or None if too few real
    detections. `compute_features`/`compute_astro_object` keep the named,
    NaN-inclusive frame for classification.

    `version` mirrors production: it is the single `feature-step` package
    version (`version("feature-step")`), mapped to a smallint via the fixture's
    FEATURE_VERSION_LUT — NOT the per-module ao.features["version"] column.
    Override `version_name` for tests.
    """
    ao = compute_astro_object(message, references_db, allwise, min_detections,
                              preprocessor=preprocessor, extractor=extractor)
    if ao is None:
        return None

    lut = feature_name_lut if feature_name_lut is not None else load_feature_name_lut()
    rows = prepare_ao_features_for_db(ao, lut)  # [name, value, band, feature_id]

    if version_name is None:
        version_name = _pkg_version("feature-step")

    rows = rows.copy()
    rows["oid"] = int(message["oid"])
    rows["sid"] = SID_ZTF
    rows["version"] = version_name_to_id(version_name)
    rows = rows.drop(columns=["name"])
    return rows[["oid", "sid", "feature_id", "band", "version", "value"]]
