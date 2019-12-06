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



class GenericMagRefTest():
    pass


class GenericXMatchTest():
    pass


class GenericMagnitudeStatisticsTest():
    pass


class GenericClassificationTest():
    pass


class GenericAstroObjectTest():
    model = None
    
    def test_get_magref(self):
        magref = self.model.get_magref()
        self.assertEqual(magref.fid, 1)

    def test_get_xmatches(self):
        xmatches = self.model.get_xmatches()
        self.assertEqual(len(xmatches),1)

    def test_get_magnitude_statistics(self):
        magstats = self.model.get_magnitude_statistics()
        self.assertEqual(magstats.fid, 1)

    def test_get_classifications(self):
        classes = self.model.get_classifications()
        print(classes)
        self.assertEqual(len(classes), 1)

class GenericFeaturesTest():
    pass


class GenericNonDetectionTest():
    pass


class GenericDetectionTest():
    pass
