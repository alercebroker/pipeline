import os
from apf.core import get_class


def multi_scale_stamp_classifier(model_class: str):
    return {
        "PARAMS": {
            "model_path": os.getenv("MODEL_PATH"),
        },
        "CLASS": model_class,
        "CLASS_MAPPER": os.getenv("CLASS_MAPPER"),
        "NAME": model_class.split(".")[-1],
    }


def configurator(model_class: str):
    if model_class.endswith("StampClassifierFull"):
        return multi_scale_stamp_classifier(model_class)

    raise Exception("Model class not found")
