import pathlib
import random
import unittest
from typing import List

import pandas as pd
from fastavro import schema, utils
from lc_classification.core.utils.no_class_post_processor import (
    NoClassifiedPostProcessor,
)
from tests.mockdata.extra_felds import generate_extra_fields
from tests.mockdata.mock_data_for_utils import (
    complete_classifications_df,
    incomplete_classifications_df,
    messages_df,
)


def generate_messages_elasticc() -> List[dict]:
    schema_path = pathlib.Path(
        pathlib.Path(__file__).parent.parent.parent.parent,
        "schemas/feature_step",
        "output.avsc",
    )
    schema_path = str(schema_path)
    input_schema = schema.load_schema(schema_path)
    messages_elasticc = list(utils.generate_many(input_schema, 2))

    for idx, message in enumerate(messages_elasticc):
        message["oid"] = f"oid{idx+1}"
        for det in message["detections"]:
            det["oid"] = message["oid"]
            det["candid"] = random.randint(0, 100000)
            det["extra_fields"] = generate_extra_fields()
        message["detections"][0]["new"] = True
        message["detections"][0]["has_stamp"] = True
    return messages_elasticc


def generate_messages_ztf() -> List[dict]:
    schema_path = (
        pathlib.Path(__file__).parent.parent.parent.parent
        / "schemas"
        / "feature_step"
        / "output.avsc"
    )
    schema_path = str(schema_path)
    input_schema = schema.load_schema(schema_path)
    messages_ztf = list(utils.generate_many(input_schema, 2))

    for message in messages_ztf:
        for det in message["detections"]:
            det["oid"] = message["oid"]
            det["extra_fields"] = {}
        message["detections"][0]["new"] = True
        message["detections"][0]["has_stamp"] = True
    return messages_ztf


class NoClassifiedPostProcessorTestCase(unittest.TestCase):
    def test_all_oid_classified(self):
        expected_df = pd.DataFrame(
            [
                [0.1, 0.2, 0.7, 0],
                [0.3, 0.1, 0.6, 0],
                [0.8, 0.1, 0.1, 0],
                [0.2, 0.5, 0.3, 0],
                [0.6, 0.2, 0.2, 0],
            ],
            index=[
                "oid1",
                "oid2",
                "oid3",
                "oid4",
                "oid5",
            ],
            columns=["class1", "class2", "class3", "NotClassified"],
        )
        expected_df.index.name = "oid"

        procesor = NoClassifiedPostProcessor(
            messages_df, complete_classifications_df
        )
        result_df = procesor.get_modified_classifications()

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_some_oid_not_classified(self):
        expected_df = pd.DataFrame(
            [
                [0.1, 0.2, 0.7, 0],
                [0, 0, 0, 1],
                [0.8, 0.1, 0.1, 0],
                [0, 0, 0, 1],
                [0.6, 0.2, 0.2, 0],
            ],
            index=[
                "oid1",
                "oid2",
                "oid3",
                "oid4",
                "oid5",
            ],
            columns=["class1", "class2", "class3", "NotClassified"],
        )
        expected_df.index.name = "oid"

        procesor = NoClassifiedPostProcessor(
            messages_df, incomplete_classifications_df
        )
        result_df = procesor.get_modified_classifications()

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_unordered_data(self):
        pass
