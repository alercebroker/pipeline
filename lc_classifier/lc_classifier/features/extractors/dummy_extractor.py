from ..core.base import FeatureExtractor, AstroObject

class DummyExtractor(FeatureExtractor):
    def compute_features_single_object(self, astro_object: AstroObject):
        print(" Dummy Extractor called ")
        #modificar astrobject
        return None