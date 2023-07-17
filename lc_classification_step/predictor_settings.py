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


def configurator(predictor_class: str):
    if predictor_class.endswith("BaltoPredictor"):
        return balto_params()
    if predictor_class.endswith("MessiPredictor"):
        return messi_params()
    if predictor_class.endswith("TorettoPredictor"):
        return toretto_params()
    if predictor_class.endswith("BarneyPredictor"):
        return barney_params()
