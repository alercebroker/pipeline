from tests.mockdata.input_elasticc import INPUT_SCHEMA as INPUT_ELASTICC
from tests.mockdata.extra_felds import generate_extra_fields
from fastavro import utils
import pytest
import os


messages_elasticc = list(utils.generate_many(INPUT_ELASTICC, 2))
for message in messages_elasticc:
    for det in message["detections"]:
        det["aid"] = message["aid"]
        det["extra_fields"] = generate_extra_fields()
    message["detections"][0]["new"] = True
    message["detections"][0]["has_stamp"] = True


@pytest.mark.skipif(os.getenv("STREAM") != "elasticc", reason="elasticc only")
def test_step_balto(test_elasticc_model, step_factory_balto):
    test_elasticc_model(step_factory_balto, messages_elasticc)
