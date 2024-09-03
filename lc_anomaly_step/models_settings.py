import os
from apf.core import get_class


def balto_params(model_class: str):
    return {
        "CLASS": model_class,
        "NAME": model_class.split(".")[-1],
        "PARAMS": {
            "model_path": os.getenv("MODEL_PATH"),
            "header_quantiles_path": os.getenv("QUANTILES_PATH"),
            "mapper": os.getenv("MAPPER_CLASS"),
        },
    }


def messi_params(model_class: str):
    return {
        "CLASS": model_class,
        "NAME": model_class.split(".")[-1],
        "PARAMS": {
            "model_path": os.getenv("MODEL_PATH"),
            "header_quantiles_path": os.getenv("HEADER_QUANTILES_PATH"),
            "feature_quantiles_path": os.getenv("FEATURE_QUANTILES_PATH"),
            "mapper": os.getenv("MAPPER_CLASS"),
        },
    }


def barney_params(model_class: str):
    return {
        "CLASS": model_class,
        "NAME": model_class.split(".")[-1],
        "PARAMS": {
            "path_to_model": os.getenv("MODEL_PATH"),
            "mapper": os.getenv("MAPPER_CLASS"),
        },
    }


def toretto_params(model_class: str):
    return {
        "CLASS": model_class,
        "NAME": model_class.split(".")[-1],
        "PARAMS": {
            "path_to_model": os.getenv("MODEL_PATH"),
            "mapper": os.getenv("MAPPER_CLASS"),
        },
    }


def ztf_params(model_class: str):
    return {
        "CLASS": model_class,
        "NAME": "lc_classifier",
    }


def anomaly_params(model_class: str):
    return {
        "PARAMS": {
            "model_path": os.getenv("MODEL_PATH"),
            "feature_quantiles_path": os.getenv("FEATURE_QUANTILES_PATH"),
            "mapper": os.getenv("MAPPER_CLASS"),
        },
        "CLASS": model_class,
        "NAME": model_class.split(".")[-1],
    }


def mbappe_params(model_class: str):
    return {
        "PARAMS": {
            "model_path": os.getenv("MODEL_PATH"),
            "features_quantiles_path": os.getenv("FEATURE_QUANTILES_PATH"),
            "metadata_quantiles_path": os.getenv("METADATA_QUANTILES_PATH"),
            "mapper": os.getenv("MAPPER_CLASS"),
        },
        "CLASS": model_class,
        "NAME": model_class.split(".")[-1],
    }


def squidward_params(model_class: str):

    return {
        "PARAMS": {
            "model_path": os.getenv("MODEL_PATH"),
            "mapper": os.getenv("MAPPER_CLASS"),
        },
        "CLASS": model_class,
        "NAME": model_class.split(".")[-1],
    }


def configurator(model_class: str):
    if model_class.endswith("BaltoClassifier"):
        return balto_params(model_class)
    if model_class.endswith("MessiClassifier"):
        return messi_params(model_class)
    if model_class.endswith("RandomForestFeaturesClassifier"):
        return toretto_params(model_class)
    if model_class.endswith("AnomalyDetector"):
        return anomaly_params(model_class)
    if model_class.endswith("MbappeClassifier"):
        return mbappe_params(model_class)
    if model_class.endswith("SquidwardFeaturesClassifier"):
        return squidward_params(model_class)
    if model_class.endswith(
        "RandomForestFeaturesHeaderClassifier"
    ) or model_class.endswith("TinkyWinkyClassifier"):
        return barney_params(model_class)
    if model_class.endswith("HierarchicalRandomForest"):
        return ztf_params(model_class)
    raise Exception("Model class not found")
