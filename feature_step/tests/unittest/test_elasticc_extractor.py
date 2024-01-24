import unittest
from features.core.elasticc import ELAsTiCCFeatureExtractor
from tests.data.elasticc_message_factory import (
    generate_input_batch,
    generate_bad_emag_ratio,
    ELASTICC_BANDS,
)


class TestElasticcExtractor(unittest.TestCase):
    def test_extractor(self):
        messages = generate_input_batch(5, ELASTICC_BANDS)

        detections = []

        for message in messages:
            detections.extend(message.get("detections", []))

        extractor = ELAsTiCCFeatureExtractor(
            detections=detections,
            non_detections=[],
            xmatch=[],
        )
        result = extractor.generate_features()

        self.assertEquals(result.shape, (5, 498))

    def test_border_case_non_present_bands(self):
        """
        If there are many objects and some object have alerts in some bands but not other
        that where present in other objects, the result of calculating the features raises
        an error. (border case resolved)
        """
        bands_incomplete = ["r", "i", "z", "Y"]
        bands_incomplete_2 = ["u", "g"]
        messages = generate_input_batch(
            5, bands_incomplete
        ) + generate_input_batch(5, bands_incomplete_2, offset=10)

        detections = []

        for message in messages:
            detections.extend(message.get("detections", []))

        extractor = ELAsTiCCFeatureExtractor(
            detections=detections,
            non_detections=[],
            xmatch=[],
        )
        result = extractor.generate_features()

        self.assertEquals(result.shape, (10, 498))

    def test_border_case_non_detected_detections(self):
        """
        There is a check to evaluate de correctnes of a detection in an object
        If theres is no correctly detected detection, the features still advance
        and at some point (calculating the sn features in particular) raises an
        error. (border case resolved)
        """
        messages = generate_bad_emag_ratio(1, ELASTICC_BANDS)

        detections = []

        for message in messages:
            detections.extend(message.get("detections", []))

        extractor = ELAsTiCCFeatureExtractor(
            detections=detections,
            non_detections=[],
            xmatch=[],
        )
        result = extractor.generate_features()

        self.assertEquals(result.shape, (1, 498))

    def test_extractor_shuffled_bands(self):
        shuffled_bands = ["r", "i", "u", "g", "Y", "z"]
        messages = generate_input_batch(1, shuffled_bands)

        detections = []

        for message in messages:
            detections.extend(message.get("detections", []))

        extractor = ELAsTiCCFeatureExtractor(
            detections=detections,
            non_detections=[],
            xmatch=[],
        )
        result = extractor.generate_features()

        self.assertEquals(result.shape, (1, 498))


class TestElasticcExtractorColorBordercases(unittest.TestCase):
    def test_border_case_colors_missing_1(self):
        bands = ["u"]
        messages = generate_input_batch(1, bands)

        detections = []

        for message in messages:
            detections.extend(message.get("detections", []))

        extractor = ELAsTiCCFeatureExtractor(
            detections=detections,
            non_detections=[],
            xmatch=[],
        )
        result = extractor.generate_features()

        self.assertEquals(result.shape, (1, 498))

    def test_border_case_colors_missing_2(self):
        bands = ["u", "g"]
        messages = generate_input_batch(1, bands)

        detections = []

        for message in messages:
            detections.extend(message.get("detections", []))

        extractor = ELAsTiCCFeatureExtractor(
            detections=detections,
            non_detections=[],
            xmatch=[],
        )
        result = extractor.generate_features()

        self.assertEquals(result.shape, (1, 498))

    def test_border_case_colors_missing_3(self):
        bands = ["g", "r", "i", "z", "Y"]
        messages = generate_input_batch(1, bands)

        detections = []

        for message in messages:
            detections.extend(message.get("detections", []))

        extractor = ELAsTiCCFeatureExtractor(
            detections=detections,
            non_detections=[],
            xmatch=[],
        )
        result = extractor.generate_features()

        self.assertEquals(result.shape, (1, 498))

    def test_border_case_colors_missing_4(self):
        bands = ["r", "i", "z"]
        messages = generate_input_batch(1, bands)

        detections = []

        for message in messages:
            detections.extend(message.get("detections", []))

        extractor = ELAsTiCCFeatureExtractor(
            detections=detections,
            non_detections=[],
            xmatch=[],
        )
        result = extractor.generate_features()

        self.assertEquals(result.shape, (1, 498))

    def test_border_case_colors_missing_5(self):
        bands = ["u", "g", "r", "i", "z"]
        messages = generate_input_batch(1, bands)

        detections = []

        for message in messages:
            detections.extend(message.get("detections", []))

        extractor = ELAsTiCCFeatureExtractor(
            detections=detections,
            non_detections=[],
            xmatch=[],
        )
        result = extractor.generate_features()

        self.assertEquals(result.shape, (1, 498))
