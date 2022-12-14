import abc
from math import ceil
from typing import Type


class DatabaseCreator(abc.ABC):
    """Abstract DatabaseConnection creator class.

    The Creator class declares the factory method that is supposed to return an
    object of a Product class. The Creator's subclasses usually provide the
    implementation of this method.
    """

    @classmethod
    @abc.abstractmethod
    def create_database(cls):
        """Abstract factory method.

        Note that the Creator may also provide some default implementation of
        the factory method.
        """
        raise NotImplementedError()


class DatabaseConnection(abc.ABC):
    """Abstract DatabaseConnection class.

    Main database connection interface declares common functionality
    to all databases
    """

    @abc.abstractmethod
    def connect(self, config):
        """Initiate the database connection."""
        raise NotImplementedError()

    @abc.abstractmethod
    def create_db(self):
        """Create database collections or tables."""
        raise NotImplementedError()

    @abc.abstractmethod
    def drop_db(self):
        """Remove database collections or tables."""
        raise NotImplementedError()

    @abc.abstractmethod
    def query(self, *args):
        """Get a query object."""
        raise NotImplementedError()


def new_DBConnection(creator: Type[DatabaseCreator]) -> DatabaseConnection:
    """Create a new database connection.

    The client code works with an instance of a concrete creator,
    albeit through its base interface. As long as the client keeps working with
    the creator via the base interface, you can pass it any creator's subclass.
    """
    return creator.create_database()


class BaseQuery(abc.ABC):
    """Abstract Query class."""

    @abc.abstractmethod
    def check_exists(self, model, filter_by):
        """Check if a model exists in the database."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_or_create(self, model, filter_by, **kwargs):
        """Create a model if it doesn't exist in the database.

        It always returns a model instance and whether it was created or not.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, instance, args):
        """Update a model instance with specified args."""
        raise NotImplementedError()

    @abc.abstractmethod
    def paginate(self, page=1, per_page=10, count=True):
        """Return a pagination object from this query."""
        raise NotImplementedError()

    @abc.abstractmethod
    def bulk_insert(self, objects, model):
        """Insert multiple objects to the database."""
        raise NotImplementedError()

    @abc.abstractmethod
    def find_all(self):
        """Retrieve all items from the result of this query."""
        raise NotImplementedError()

    @abc.abstractmethod
    def find_one(self, filter_by={}, model=None, **kwargs):
        """Retrieve only one item from the result of this query.

        Returns None if result is empty.
        """
        raise NotImplementedError()


class Pagination:
    """Paginate responses from the database."""

    def __init__(self, query, page, per_page, total, items):
        """Set attributes from args."""
        self.query = query
        self.page = page
        self.per_page = per_page
        self.total = total
        self.items = items

    @property
    def pages(self):
        """Get total number of pages."""
        if self.per_page == 0 or self.total is None:
            pages = 0
        else:
            pages = int(ceil(self.total / float(self.per_page)))
        return pages

    def prev(self):
        """Return a :class:`Pagination` object for the previous page."""
        assert (
            self.query is not None
        ), "a query object is required for this method to work"
        return self.query.paginate(self.page - 1, self.per_page)

    @property
    def prev_num(self):
        """Get number of the previous page."""
        if not self.has_prev:
            return None
        return self.page - 1

    @property
    def has_prev(self):
        """Check if a previous page exists."""
        return self.page > 1

    def next(self):
        """Return a :class:`Pagination` object for the next page."""
        assert (
            self.query is not None
        ), "a query object is required for this method to work"
        return self.query.paginate(self.page + 1, self.per_page)

    @property
    def has_next(self):
        """Check if a next page exists."""
        return self.page < self.pages

    @property
    def next_num(self):
        """Get number of the next page."""
        if not self.has_next:
            return None
        return self.page + 1


class PaginationNoCount(Pagination):
    def __init__(self, query, page, per_page, items, has_next):
        super().__init__(query, page, per_page, None, items)
        self._has_next = has_next

    @property
    def has_next(self):
        """Check if a previous page exists."""
        return self._has_next


class AbstractObject:
    """Abstract Object model.

    Attributes
    ----------
    oid: str
        object identifier
    nobs: str
        number of observations
    lastmjd: float
        date (in mjd) of the last observation
    firstmjd: float
        date (in mjd) of the first observation
    meanra: float
        mean right ascension coordinates
    meandec: float
        mean declination coordinates
    sigmara: float
        error for right ascension coordinates
    sigmadec: float
        error for declination coordinates
    deltajd: float
        difference between last and first mjd
    """

    oid = None
    ndethist = None
    ncovhist = None
    mjdstarthist = None
    mjdendhist = None
    corrected = None
    stellar = None
    ndet = None
    g_r_max = None
    g_r_max_corr = None
    g_r_mean = None
    g_r_mean_corr = None
    meanra = None
    meandec = None
    sigmara = None
    sigmadec = None
    deltajd = None
    firstmjd = None
    lastmjd = None
    step_id_corr = None

    def get_lightcurve(self):
        """Get the lightcurve of the object."""
        pass


class AbstractMagnitudeStatistics:
    """Abstract Magnitude Statistics model.

    Attributes
    ----------
    magnitude_type : str
        the type of the magnitude, could be psf or ap
    fid : int
        magnitude band identifier, 1 for red, 2 for green
    mean : float
        mean magnitude meassured
    median : float
        median of the magnitude
    max_mag : float
        maximum value of magnitude meassurements
    min_mag : float
        minimum value of magnitude meassurements
    sigma : float
        error of magnitude meassurements
    last : float
        value for the last magnitude meassured
    first : float
        value for the first magnitude meassured
    """

    oid = None
    fid = None
    stellar = None
    corrected = None
    ndet = None
    ndubious = None
    dmdt_first = None
    dm_first = None
    sigmadm_first = None
    dt_first = None
    magmean = None
    magmedian = None
    magmax = None
    magmin = None
    magsigma = None
    maglast = None
    magfirst = None
    firstmjd = None
    lastmjd = None
    step_id_corr = None


class AbstractNonDetection:
    """Abstract model for non detections.

    Attributes
    ----------
    mjd : float
        date of the non detection in mjd
    diffmaglim: float
        magnitude of the non detection
    fid : int
        band identifier 1 for red, 2 for green
    """

    mjd = None
    diffmaglim = None
    fid = None


class AbstractDetection:
    """Abstract model for detections.

    Attributes
    ----------
    candid : str
        candidate identifier
    mjd : float
        date of the detection in mjd
    fid : int
        band identifier, 1 for red, 2 for green
    ra : float
        right ascension coordinates
    dec : float
        declination coordinates
    rb : int
        real bogus
    magap : float
        ap magnitude
    magpsf : float
        psf magnitude
    sigmapsf : float
        error for psf magnitude
    sigmagap : float
        error for ap magnitude
    magpsf_corr : float
        magnitude correction for magpsf
    magap_corr : float
        magnitude correction for magap
    sigmapsf_corr : float
        correction for sigmapsf
    sigmagap_corr : float
        correction for sigmagap
    avro : string
        url for avro file in s3
    """

    avro = None
    oid = None
    candid = None
    mjd = None
    fid = None
    pid = None
    diffmaglim = None
    isdiffpos = None
    nid = None
    ra = None
    dec = None
    magpsf = None
    sigmapsf = None
    magap = None
    sigmagap = None
    distnr = None
    rb = None
    rbversion = None
    drb = None
    drbversion = None
    magapbig = None
    sigmagapbig = None
    rfid = None
    magpsf_corr = None
    sigmapsf_corr = None
    sigmapsf_corr_ext = None
    corrected = None
    dubious = None
    parent_candid = None
    has_stamp = None
    step_id_corr = None


class AbstractDataquality:
    """Abstract Dataquality model."""

    candid = None
    oid = None
    fid = None
    xpos = None
    ypos = None
    chipsf = None
    sky = None
    fwhm = None
    classtar = None
    mindtoedge = None
    seeratio = None
    aimage = None
    bimage = None
    aimagerat = None
    bimagerat = None
    nneg = None
    nbad = None
    sumrat = None
    scorr = None
    dsnrms = None
    ssnrms = None
    magzpsci = None
    magzpsciunc = None
    magzpscirms = None
    nmatches = None
    clrcoeff = None
    clrcounc = None
    zpclrcov = None
    zpmed = None
    clrmed = None
    clrrms = None
    exptime = None


class AbstractGaia_ztf:
    """Abstract Gaia model."""

    oid = None
    candid = None
    neargaia = None
    neargaiabright = None
    maggaia = None
    maggaiabright = None
    unique = None


class AbstractSs_ztf:
    """Abstract ss_ztf model."""

    oid = None
    candid = None
    ssdistnr = None
    ssmagnr = None
    ssnamenr = None


class AbstractPs1_ztf:
    """Abstract ps1_ztf model."""

    oid = None
    candid = None
    objectidps1 = None
    sgmag1 = None
    srmag1 = None
    simag1 = None
    szmag1 = None
    sgscore1 = None
    distpsnr1 = None
    objectidps2 = None
    sgmag2 = None
    srmag2 = None
    simag2 = None
    szmag2 = None
    sgscore2 = None
    distpsnr2 = None
    objectidps3 = None
    sgmag3 = None
    srmag3 = None
    simag3 = None
    szmag3 = None
    sgscore3 = None
    distpsnr3 = None
    nmtchps = None
    unique1 = None


class AbstractReference:
    """Abstract reference model."""

    oid = None
    rfid = None
    candiid = None
    fid = None
    rcid = None
    field = None
    magnr = None
    sigmagnr = None
    chinr = None
    sharpnr = None
    ranr = None
    decnr = None
    mjdstartref = None
    mjdendref = None
    nframesref = None


class AbstractPipeline:
    """Abstract pipeline model."""

    pipeline_id = None
    step_id_corr = None
    step_id_feat = None
    step_id_clf = None
    step_id_out = None
    step_id_stamp = None
    date = None


class AbstractStep:
    """Abstract Step model."""

    step_id = None
    name = None
    version = None
    comments = None
    date = None
