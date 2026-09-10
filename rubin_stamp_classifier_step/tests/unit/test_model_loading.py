"""The step builds its model from MODEL_CONFIG.CLASS and MODEL_CONFIG.PARAMS.

Nothing in the step names a concrete model class: the rubin and hunter
deployments differ only in config.
"""
from unittest import mock

import pytest

from rubin_stamp_classifier_step.step import StampClassifierStep
from tests.unit.stub_model import CLASSES, StubModel

STUB_CLASS = "tests.unit.stub_model.StubModel"
TAXONOMY = {name: idx + 10 for idx, name in enumerate(CLASSES)}


def build_step(model_config):
    config = {
        "CONSUMER_CONFIG": {"CLASS": "apf.core.step.DefaultConsumer"},
        "PRODUCER_CONFIG": {"CLASS": "apf.core.step.DefaultProducer"},
        "DB_CONFIG": {},
        "MODEL_CONFIG": model_config,
    }
    with mock.patch("rubin_stamp_classifier_step.step.PSQLConnection"), \
         mock.patch("rubin_stamp_classifier_step.step.get_taxonomy_by_classifier_id", return_value=TAXONOMY):
        return StampClassifierStep(config=config)


def test_model_is_built_from_class_with_params_as_kwargs():
    step = build_step({
        "CLASS": STUB_CLASS,
        "PARAMS": {"model_path": "https://example.org/1.0.0/model.zip", "threshold": 0.5},
        "CLS_ID": 3,
    })

    assert isinstance(step.model, StubModel)
    assert step.model.params == {"model_path": "https://example.org/1.0.0/model.zip", "threshold": 0.5}
    assert step.dict_mapping_classes == step.model.dict_mapping_classes


def test_missing_params_means_no_kwargs():
    step = build_step({"CLASS": STUB_CLASS, "CLS_ID": 3})

    assert step.model.params == {}


def test_missing_class_is_a_config_error():
    with pytest.raises(KeyError, match="MODEL_CONFIG.CLASS"):
        build_step({"PARAMS": {"model_path": "x"}, "CLS_ID": 3})
