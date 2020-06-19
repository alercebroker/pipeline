class GenericClassTest():
    pass


class GenericTaxonomyTest():
    pass


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

    def test_get_lightcurve(self):
        light_curve = self.model.get_lightcurve()
        self.assertIsInstance(light_curve, dict)
        self.assertTrue("detections" in light_curve)
        self.assertTrue("non_detections" in light_curve)
        self.assertIsInstance(light_curve["detections"], list)
        self.assertIsInstance(light_curve["non_detections"], list)


class GenericFeaturesTest():
    pass


class GenericNonDetectionTest():
    pass


class GenericDetectionTest():
    pass
