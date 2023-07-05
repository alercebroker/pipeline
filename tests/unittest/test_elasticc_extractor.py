import unittest
import json
from features.core.elasticc import ELAsTiCCFeatureExtractor
from tests.data.elasticc_message_factory import generate_input_batch, generate_bad_emag_ratio, ELASTICC_BANDS


class TestElasticcExtractor(unittest.TestCase):

    def test_extractor(self):
        messages = generate_input_batch(5, ELASTICC_BANDS)
        
        detections, non_detections, xmatch = [], [], []

        for message in messages:
            detections.extend(message.get("detections", []))
            non_detections.extend(message.get("non_detections", []))
            xmatch.append({"aid": message["aid"], **(message.get("xmatches", {}) or {})})

        extractor = ELAsTiCCFeatureExtractor(
            detections,
            non_detections,
            xmatch
        )
        result = extractor.generate_features()

        self.assertEquals(result.shape, (5, 429))

    def test_border_case_non_present_bands(self):
        """
        If there are many objects and some object have alerts in some bands but not other
        that where present in other objects, the result of calculating the features raises
        an error
        """
        BANDS = ["u", "g"]
        BANDS2 = ["r", "i", "z", "Y"]
        messages = generate_input_batch(5, BANDS) + generate_input_batch(5, BANDS2, offset=10)
        
        detections, non_detections, xmatch = [], [], []

        for message in messages:
            detections.extend(message.get("detections", []))
            non_detections.extend(message.get("non_detections", []))
            xmatch.append({"aid": message["aid"], **(message.get("xmatches", {}) or {})})

        extractor = ELAsTiCCFeatureExtractor(
            detections,
            non_detections,
            xmatch
        )
        result = extractor.generate_features()

        self.assertEquals(result.shape, (10, 429))

    def test_border_case_non_detected_detections(self):
        """
        There is a check to evaluate de correctnes of a detection in an object
        If theres is no correctly detected detection, the features still advance
        and at some point (calculating the sn features in particular) raises an
        error
        """
        messages = generate_bad_emag_ratio(1, ELASTICC_BANDS)
        
        detections, non_detections, xmatch = [], [], []

        for message in messages:
            detections.extend(message.get("detections", []))
            non_detections.extend(message.get("non_detections", []))
            xmatch.append({"aid": message["aid"], **(message.get("xmatches", {}) or {})})

        extractor = ELAsTiCCFeatureExtractor(
            detections,
            non_detections,
            xmatch
        )
        result = extractor.generate_features()

        self.assertEquals(result.shape, (1, 429))
