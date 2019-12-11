class GenericClassTest():
    model = None
    def test_get_taxonomies(self):
        taxonomies = self.model.get_taxonomies()
        self.assertIsInstance(taxonomies, list)


class GenericTaxonomyTest():
    model = None
    def test_get_classes(self):
        classes = self.model.get_classes()
        self.assertIsInstance(classes, list)


class GenericClassifierTest():
    pass


class GenericXMatchTest():
    pass


class GenericMagnitudeStatisticsTest():
    pass


class GenericClassificationTest():
    pass


class GenericAstroObjectTest():
    model = None

    def test_get_xmatches(self):
        xmatches = self.model.get_xmatches()
        self.assertEqual(len(xmatches),1)

    def test_get_magnitude_statistics(self):
        magstats = self.model.get_magnitude_statistics()
        self.assertEqual(magstats.fid, 1)

    def test_get_classifications(self):
        classes = self.model.get_classifications()
        self.assertEqual(len(classes), 1)

    def test_get_features(self):
        features = self.model.get_features()
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 1)

    def test_get_detections(self):
        detections = self.model.get_detections()
        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), 1)

    def test_get_non_detections(self):
        non_detections = self.model.get_non_detections()
        self.assertIsInstance(non_detections, list)
        self.assertEqual(len(non_detections), 1)


class GenericFeaturesTest():
    pass


class GenericNonDetectionTest():
    pass


class GenericDetectionTest():
    pass
