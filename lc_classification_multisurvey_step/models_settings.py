import os


def squidward_params(model_class: str):
    """The single BHRF entry. Mirrors the stamp step's one-entry configurator so
    the model class, mapper class and model path stay env-driven (design §3)."""
    return {
        "CLASS": model_class,
        "CLASS_MAPPER": os.getenv("CLASS_MAPPER"),
        "PARAMS": {"model_path": os.getenv("MODEL_PATH")},
        "NAME": model_class.split(".")[-1],
        "VERSION": os.getenv("MODEL_VERSION", "2.1.0"),
        "CLASSIFIER_NAME": os.getenv("CLASSIFIER_NAME", "lc_classifier_BHRF_forced_phot"),
        "SID": int(os.getenv("SID", 0)),
        "MIN_DETECTIONS": (
            int(os.environ["MIN_DETECTIONS"]) if os.getenv("MIN_DETECTIONS") else None
        ),
    }


def configurator(model_class: str):
    if model_class.endswith("SquidwardFeaturesClassifier"):
        return squidward_params(model_class)

    raise Exception(f"Model class not supported by this step: {model_class}")
