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


def classify_oid(oid: int, credentials: str, model, min_detections: int = 1,
                 preprocessor=None, extractor=None):
    """DB -> message -> features -> probabilities for one oid.

    Returns an OutputDTO, or None if the object has too few real detections."""
    from features.offline import db
    from features.offline.message import build_message

    oids = [oid]
    dets = db.fetch_detections(credentials, oids)
    forced = db.fetch_forced_photometry(credentials, oids)
    ps1 = db.fetch_ps1(credentials, oids)
    allwise = db.fetch_allwise(credentials, oids)
    refs = db.fetch_references(credentials, oids)

    message = build_message(oid, dets, forced, ps1)
    ao = compute_astro_object(message, refs, allwise, min_detections,
                              preprocessor=preprocessor, extractor=extractor)
    if ao is None:
        return None
    return classify_astro_object(ao, message, model)
