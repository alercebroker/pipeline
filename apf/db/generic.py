class AbstractClass():
    # def __init__(self, name, acronym):
    #     self.name = name
    #     self.acronym = acronym
    def get_taxonomies(self):
        pass


class AbstractTaxonomy():
    def get_classes(self):
        pass

    def get_classifiers(self):
        pass


class AbstractClassifier():
    def get_features(self):
        pass

    def get_classifications(self):
        pass


class AbstractAstroObject():
    def get_classifications(self):
        pass

    def get_magnitude_statistics(self):
        pass

    def get_xmatches(self):
        pass

    def get_magref(self):
        pass

    def get_features(self):
        pass

    def get_detections(self):
        pass

    def get_non_detections(self):
        pass

class AbstractMagRef():
    