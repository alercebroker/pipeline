from tests.mockdata.input_elasticc import INPUT_SCHEMA as INPUT_ELASTICC
from fastavro import utils
import pytest
import os

messages_elasticc = list(utils.generate_many(INPUT_ELASTICC, 10))
for message in messages_elasticc:
    for det in message["detections"]:
        det["aid"] = message["aid"]
    message["detections"][0]["new"] = True
    message["detections"][0]["has_stamp"] = True


@pytest.mark.skipif(os.getenv("STREAM") != "elasticc", reason="elasticc only")
def test_step_balto(test_elasticc_model, step_factory_balto):
    test_elasticc_model(step_factory_balto, messages_elasticc)
