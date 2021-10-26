"""Abstract Model definitions."""


class Object:
    """Abstract Object class."""

    __tablename__ = "object"

    aid = None
    sid = None
    lastmjd = None
    firstmjd = None
    meanra = None
    meandec = None
    sigmara = None
    sigmadec = None


class Detection:
    """Abstract Detection class."""

    __tablename__ = "detection"

    aid = None
    sid = None
    candid = None
    mjd = None
    fid = None
    ra = None
    dec = None
    rb = None
    mag = None
    sigmag = None
    extra_fields = None


class NonDetection:
    """Abstract NonDetection class."""

    __tablename__ = "non_detection"

    aid = None
    sid = None
    mjd = None
    diffmaglim = None
    fid = None
    extra_fields = None


class Classification:
    """Abstract Classification class."""

    __tablename__ = "classification"

    alerce_id = None
    survey_id = None
    classifier_name = None
    classifier_version = None
    class_name = None
    probabilty = None
    ranking = None
    probabilities = None
    last_updated = None
    created_on = None


class Features:
    """Abstract Features class."""

    __tablename__ = "features"

    alerce_id = None
    survey_id = None
    features = None
    created_on = None
    last_updated = None


class Xmatch:
    """Abstract Xmatch class."""

    __tablename__ = "xmatch"

    alerce_id = None
    survey_id = None
    catalog_name = None
    catalog_id = None
