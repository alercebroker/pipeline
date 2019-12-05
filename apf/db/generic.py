class AbstractClass():
    id = None
    name = None
    acronym = None

    def get_taxonomies(self):
        pass


class AbstractTaxonomy():
    id = None
    name = None
    def get_classes(self):
        pass

    def get_classifiers(self):
        pass


class AbstractClassifier():
    id = None
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

    def get_magref(self):
        pass

    def get_features(self):
        pass

    def get_detections(self):
        pass

    def get_non_detections(self):
        pass

    def get_lightcurve(self):
        pass

class AbstractMagRef():
    id = None
    fid = None
    rcid = None
    field = None
    magref = None
    sigmagref = None
    corrected = None

class AbstractXmatch():
    catalog_id = None
    catalog_object_id = None

class AbstractMagnitudeStatistics():
    id = None
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
    id = None
    class_name = None
    probability = None

class AbstractFeatures():
    id = None
    data = None

class AbstractNonDetection():
    id = None
    mjd = None
    diffmaglim = None
    fid = None

class AbstractDetection():
    id = None
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
    