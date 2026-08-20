"""Offline classification: features AstroObject -> BHRF probabilities.

Stitches the real feature_step output parser (`parse_output`), the real
alerce_classifiers InputDTO factory, and the real SquidwardFeaturesClassifier
(BHRF) together, without the lc_classification Kafka step. The Squidward model
reads only InputDTO.features, so detections/non_detections/xmatch/stamps are
passed empty (this also sidesteps lc_classification_step's stale candid schema).

Model config is read from the same env vars the deployed step uses:
    MODEL_PATH   (required) - model pickle URL/path (e.g. the S3 BHRF 2.1.0 url)
    MAPPER_CLASS (optional) - defaults to the Squidward mapper
    CLASSIFIER_NAME (optional) - output label; deployment uses lc_classifier_BHRF_forced_phot
"""
import os

import pandas as pd
from apf.core import get_class
from alerce_classifiers.base.dto import OutputDTO
from alerce_classifiers.base.factories import input_dto_factory

from features.utils.parsers import parse_output
from features.offline import db
from features.offline import xmatch
from features.offline.message import build_message
from .lc_features import compute_astro_object

DEFAULT_MODEL_CLASS = "alerce_classifiers.squidward.model.SquidwardFeaturesClassifier"
DEFAULT_MAPPER_CLASS = "alerce_classifiers.squidward.mapper.SquidwardMapper"


def load_squidward_model(model_class: str = DEFAULT_MODEL_CLASS):
    """Instantiate the BHRF classifier from env vars (mirrors the deployed step).

    Returns (model, classifier_name, classifier_version). The version is derived
    by the model from the model path (e.g. ".../squidward/2.1.0/..." -> "2.1.0").
    """
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        raise ValueError("MODEL_PATH env var is required to load the model")
    mapper_class = os.getenv("MAPPER_CLASS", DEFAULT_MAPPER_CLASS)
    mapper = get_class(mapper_class)()
    model = get_class(model_class)(model_path=model_path, mapper=mapper)
    name = os.getenv("CLASSIFIER_NAME", model_class.split(".")[-1])
    return model, name, model.model_version


def features_message_to_dto(out_message: dict):
    """Classifier-input message -> features-only InputDTO.

    `out_message["features"]` is the wide, band-suffixed feature dict produced by
    parse_output. The model reads only features; detections/non_detections/xmatch/
    stamps are empty."""
    features = out_message.get("features") or {}
    features_df = pd.DataFrame([features], index=[out_message["oid"]])
    features_df.index.name = "oid"
    empty = pd.DataFrame()
    return input_dto_factory(empty, empty, features_df, empty, empty)


def _empty_output() -> OutputDTO:
    return OutputDTO(pd.DataFrame(), {"top": pd.DataFrame(), "children": {}})


def classify_astro_object(ao, message: dict, model) -> OutputDTO:
    """Post-extract AstroObject + its source message -> OutputDTO.

    Uses the real parse_output to name the features, builds a features-only DTO,
    then runs can_predict + predict. Returns an empty OutputDTO if the model
    can't predict (mirrors the step's behavior)."""
    candids = {message["oid"]: message.get("measurement_id", [])}
    out_message = parse_output([ao], [message], candids)[0]
    dto = features_message_to_dto(out_message)
    can, _ = model.can_predict(dto)
    if not can:
        return _empty_output()
    return model.predict(dto)


def _fetch_oid_inputs(oid: int, credentials: str, xmatch_url: str = None):
    """DB -> (message, references, allwise, detections, forced, matches) for one oid.

    AllWISE source mirrors the live step's choice:
      - `xmatch_url` set -> compute the crossmatch against Xwave (like
        `step.pre_execute`), using the message's meanra/meandec as the cone
        center; `matches` is the raw MatchWithMetadata list (for persistence).
      - `xmatch_url` unset -> read the precomputed `multisurvey_ztf.xmatch ⋈
        allwise` (empty for ZTF today); `matches` is [].
    """
    oids = [oid]
    dets = db.fetch_detections(credentials, oids)
    forced = db.fetch_forced_photometry(credentials, oids)
    ps1 = db.fetch_ps1(credentials, oids)
    refs = db.fetch_references(credentials, oids)
    message = build_message(oid, dets, forced, ps1)

    # .get: the cone centre is only needed for the live crossmatch, so the DB-read
    # path must not require the message to carry coordinates.
    allwise, matches = xmatch.allwise_for_oid(
        oid, message.get("meanra"), message.get("meandec"), credentials,
        xmatch_url=xmatch_url)
    return message, refs, allwise, dets, forced, matches


def _lc_lastmjd(dets, forced):
    """Max MJD over all epochs the classifier consumed (detections + forced).

    Already MJD (db.py reads mjd) — do NOT subtract 2400000.5. None if no epochs.
    """
    mjds = []
    if dets is not None and len(dets):
        mjds.append(float(dets["mjd"].max()))
    if forced is not None and len(forced):
        mjds.append(float(forced["mjd"].max()))
    return max(mjds) if mjds else None


def classify_oid(oid: int, credentials: str, model, min_detections: int = 1,
                 preprocessor=None, extractor=None, xmatch_url: str = None):
    """DB -> message -> features -> probabilities for one oid.

    Returns an OutputDTO, or None if the object has too few real detections.
    `xmatch_url` (or XMATCH_URL env, resolved by the caller) computes the AllWISE
    crossmatch against Xwave instead of reading the empty DB tables."""
    message, refs, allwise, _dets, _forced, _matches = _fetch_oid_inputs(
        oid, credentials, xmatch_url)
    ao = compute_astro_object(message, refs, allwise, min_detections,
                              preprocessor=preprocessor, extractor=extractor)
    if ao is None:
        return None
    return classify_astro_object(ao, message, model)


def classify_oid_for_save(oid: int, credentials: str, model, min_detections: int = 1,
                          preprocessor=None, extractor=None, xmatch_url: str = None):
    """Like classify_oid but also returns (lastmjd, matches) for persistence.

    Returns (OutputDTO, lastmjd, matches), or (None, None, []) if too few real
    detections. lastmjd = max MJD over detections + forced (see _lc_lastmjd);
    `matches` is the raw Xwave crossmatch (empty unless xmatch_url is set)."""
    message, refs, allwise, dets, forced, matches = _fetch_oid_inputs(
        oid, credentials, xmatch_url)
    ao = compute_astro_object(message, refs, allwise, min_detections,
                              preprocessor=preprocessor, extractor=extractor)
    if ao is None:
        return None, None, []
    return (classify_astro_object(ao, message, model),
            _lc_lastmjd(dets, forced), matches)
