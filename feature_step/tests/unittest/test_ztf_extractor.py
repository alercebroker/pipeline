import unittest
from features.core.ztf import ZTFFeatureExtractor
from tests.data.elasticc_message_factory import generate_input_batch


class TestZTFExtractor(unittest.TestCase):
    # this test includes an assertion of the exclusion of fp
    # remove when necesary
    def test_init(self):
        detections = []
        input_batch = generate_input_batch(10, ["g", "r"], survey="ZTF")
        for msg in input_batch:
            detections.extend(msg.get("detections", []))

        extractor = ZTFFeatureExtractor(
            detections=detections, non_detections=[], xmatch=[]
        )
        self.assertTrue(not extractor.detections._alerts["forced"].any())
