class GenericClassTest():
    model = None
    params = {}
    def test_get_taxonomies(self):
        mod = self.model(**self.params)
        taxonomies = mod.get_taxonomies()
        self.assertIsInstance(taxonomies, list)


class GenericTaxonomyTest():
    model = None
    params = {}
    def test_get_classes(self):
        mod = self.model(**self.params)
        classes = mod.get_classes()
        self.assertIsInstance(classes, list)


class GenericClassifierTest():
    pass



class GenericMagRefTest():
    pass


class GenericXMatchTest():
    pass


class GenericMagnitudeStatisticsTest():
    pass


class GenericClassificationTest():
    pass


class GenericAstroObjectTest():
    pass


class GenericFeaturesTest():
    pass


class GenericNonDetectionTest():
    pass


class GenericDetectionTest():
    pass
