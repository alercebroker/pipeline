class AbstractClass():
    """
    Abstract Class model

    Attributes
    ----------
    name : str
        the class name
    acronym : str
        the short class name
    """
    name = None
    acronym = None

    def get_taxonomies(self):
        """
        Gets the taxonomies that the class instance belongs to
        """
        pass


class AbstractTaxonomy():
    name = None
    def get_classes(self):
        pass

    def get_classifiers(self):
        pass


class AbstractClassifier():
    name = None

    def get_features(self):
        pass

    def get_classifications(self):
        pass


class AbstractAstroObject():
    oid = None
    nobs = None
    lastmjd = None
    meanra = None
    meandec = None
    sigmara = None
    sigmadec = None
    deltajd = None
    firstmjd = None

    def get_classifications(self):
        pass

    def get_magnitude_statistics(self):
        pass

    def get_xmatches(self):
        pass

    def get_features(self):
        pass

    def get_detections(self):
        pass

    def get_non_detections(self):
        pass

    def get_lightcurve(self):
        pass


class AbstractXmatch():
    catalog_id = None
    catalog_object_id = None

class AbstractMagnitudeStatistics():
    magnitude_type = None
    fid = None
    mean = None
    median = None
    max_mag = None
    min_mag = None
    sigma = None
    last = None
    first = None

class AbstractClassification():
    class_name = None
    probability = None

class AbstractFeatures():
    version = None

class AbstractNonDetection():
    mjd = None
    diffmaglim = None
    fid = None

class AbstractDetection():
    candid = None
    mjd = None
    fid = None
    ra = None
    dec = None
    rb = None
    magap = None
    magpsf = None
    sigmapsf = None
    sigmagap = None
    alert = None
    