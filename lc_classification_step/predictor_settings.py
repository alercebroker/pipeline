import os


def balto_params():
    return {
        "model_path": os.getenv("MODEL_PATH"),
        "quantiles_path": os.getenv("QUANTILES_PATH"),
    }


def messi_params():
    return {
        "model_path": os.getenv("MODEL_PATH"),
        "header_quantiles_path": os.getenv("HEADER_QUANTILES_PATH"),
        "feature_quantiles_path": os.getenv("FEATURE_QUANTILES_PATH"),
    }


def barney_params():
    return {
        "model_path": os.getenv("MODEL_PATH"),
    }


def toretto_params():
    return {
        "model_path": os.getenv("MODEL_PATH"),
    }


def configurator(model_class: str):
    if model_class.endswith("Balto"):
        return balto_params()
    if model_class.endswith("Messi"):
        return messi_params()
    if model_class.endswith("Toretto"):
        return toretto_params()
    if model_class.endswith("Barney"):
        return barney_params()
