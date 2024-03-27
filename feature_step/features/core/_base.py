import logging
import abc
from typing import Any

import pandas as pd
from scipy import stats

from .handlers import DetectionsHandler, NonDetectionsHandler
from .utils import decorators, extras, functions


class BaseFeatureExtractor(abc.ABC):
    r"""Base class for feature extractors.

    There are special reserved methods that start with `_PREFIX`. Do not add methods that begin with `_PREFIX`
    unless they are intended to compute features. Instances normally use the method `generate_features` to compute
    all features. Note that it is possible to exclude specific features by passing them as the `exclude` argument.
    The elements passed to exclude are the name of the feature producer (i.e., the name of the method, with or
    without `_PREFIX`).

    The extractors included (and computed) here by default are:

    * `periods`: Period related features (multi-band period, per band period, etc.). Check the notes.
    * `fats`: Computes all FATS features defined in `FATS_FEATURES`.
    * `mhps`: Mexican hat power spectrum for each object, per band.
    * `gp_drw`: Gaussian process damp random walk parameters for each object and band.
    * `iqr`: Inter-quartile range in magnitude distribution of each object, per band.

    Most settings are defined as class attributes. Additional notes:

    * `EXTRA_COLUMNS` should only include columns not included by default in the class `DetectionsHandler`.
    * If `N_HARMONICS` is 0, it will skip calculation of harmonic magnitudes and phases altogether.
    * If `POWER_RATE_FACTORS` is empty, it will skip the power rate calculations.

    Notes:
        If `periods` is excluded, `COMPUTE_KIM`, `N_HARMONICS` and `POWER_RATE_FACTORS` will all be ignored, as they
        depend on values computed in the main period calculations.

    Attributes:
        _PREFIX: Prefix of methods considered as feature extractors. It is recommended to keep it for all subclasses
        _AUTO_EXCLUDE: Feature extraction methods always excluded from calculations. Default empty
        NAME: Name of feature extractor. Must be defined in subclass
        VERSION: Version of feature extractor. Must be defined in subclass
        SURVEYS: Surveys to include in feature calculation. Empty tuple (default) means use all
        BANDS: Bands to include in feature calculation. Empty tuple (default) means use all
        BANDS_MAPPING: Conversion of bands to values used in features. Mostly for compatibility with old classifiers
        EXTRA_COLUMNS: Additional fields used for detections. Default empty
        XMATCH_COLUMNS: Mapping from cross-match catalogue to columns required in that catalogue. Default empty
        CORRECTED: Use corrected magnitudes if first detection of object is corrected. Default `False`
        FLUX: Flag to indicate that "magnitude" fields are actually flux (field names do not change). Default `False`
        COMPUTE_KIM: Compute parameters from phase-folded light-curve. Default `True`
        N_HARMONICS: Number of harmonic series parameters to compute. Default 7
        POWER_RATE_FACTORS: Ratio of power of the periodogram to the best period. Default 1/4, 1/3, 1/2, 2, 3 and 4
        MIN_DETS: Minimum number of overall detections per object to compute features. Default 0
        MIN_DETS_FID: Minimum number of detections in a band per object ro compute features. Default 5
        T1: Low frequency timescale (in days) for MHPS calculations. Default 100
        T2: High frequency timescale (in days) for MHPS calculations. Default 10
        FATS_FEATURES: FATS features to be computed. Check defaults in code
    """

    NAME: str
    VERSION: str
    _PREFIX: str = "calculate_"
    _AUTO_EXCLUDE: set[str] = set()
    SURVEYS: tuple[str, ...] = ()
    BANDS: tuple[str, ...] = ()
    BANDS_MAPPING: dict[str, Any] = {}
    EXTRA_COLUMNS: list[str] = []
    XMATCH_COLUMNS: dict[str, list] = {}
    CORRECTED: bool = False
    FLUX: bool = False
    COMPUTE_KIM: bool = True
    N_HARMONICS: int = 7
    # Elements can only be composed of numbers and "/"
    POWER_RATE_FACTORS: tuple[str, ...] = ("1/4", "1/3", "1/2", "2", "3", "4")
    MIN_DETS: int = 5
    MIN_DETS_FID: int = 0
    T1: float = 100
    T2: float = 10
    FATS_FEATURES: tuple[str, ...] = (
        "Amplitude",
        "AndersonDarling",
        "Autocor_length",
        "Beyond1Std",
        "Con",
        "Eta_e",
        "Gskew",
        "MaxSlope",
        "Mean",
        "Meanvariance",
        "MedianAbsDev",
        "MedianBRP",
        "PairSlopeTrend",
        "PercentAmplitude",
        "Q31",
        "Rcs",
        "Skew",
        "SmallKurtosis",
        "Std",
        "StetsonK",
        "Pvar",
        "ExcessVar",
        "SF_ML_amplitude",
        "SF_ML_gamma",
        "IAR_phi",
        "LinearTrend",
    )

    def __init__(
        self,
        detections: list[dict] | pd.DataFrame,
        non_detections: list[dict] | pd.DataFrame = None,
        xmatches: list[dict] | pd.DataFrame = None,
        **kwargs,
    ):
        """Initialize feature extractor.

        Detections, non-detections and cross-matches can come from multiple objects. Features are computed per object.

        Args:
            detections: All detections used for feature computing
            non_detections: All non-detections. Non-detections from objects not present in detections will be removed
            xmatches: Object cross-matches. It will be matched to detections based on its ID
        """
        self.logger = logging.getLogger(f"alerce.{self.__class__.__name__}")

        self.logger.info("Initializing feature extractor")
        common = dict(surveys=self.SURVEYS, bands=self.BANDS)

        if isinstance(detections, pd.DataFrame):
            detections = detections.reset_index().to_dict("records")
        self.detections = DetectionsHandler(
            detections,
            extras=self.EXTRA_COLUMNS,
            corr=self.CORRECTED,
            **common,
        )

        self.logger.info(
            f"Total objects before clearing: {self.detections.ids().size}"
        )
        self._discard_detections()
        self.logger.info(
            f"Total objects after clearing: {self.detections.ids().size}"
        )

        first_mjd = functions.fill_index(
            self.detections.agg("mjd", "min", by_fid=True), fid=self.BANDS
        )

        non_detections = non_detections if non_detections is not None else []
        if isinstance(non_detections, pd.DataFrame):
            non_detections = non_detections.reset_index().to_dict("records")
        self.non_detections = NonDetectionsHandler(
            non_detections, first_mjd=first_mjd, **common
        )
        self.non_detections.match(self.detections)

        if isinstance(xmatches, pd.DataFrame):
            xmatches = xmatches.reset_index().to_dict("records")
        self.xmatches = self._create_xmatches(
            xmatches or []
        )  # If None, use empty list
        if kwargs:
            raise ValueError(f"Unrecognized kwargs: {', '.join(kwargs)}")
        self.logger.info("Finished initialization")

    @abc.abstractmethod
    def _discard_detections(self):
        """Remove objects based on the minimum number of detections. Should be called with `super` in subclass
        implementations."""
        self.detections.not_enough(self.MIN_DETS)
        self.detections.not_enough(self.MIN_DETS_FID, by_fid=True)

    def _create_xmatches(self, xmatches: list[dict]) -> pd.DataFrame:
        """Ensures cross-matches contain `oid` in detections and selects required columns."""

        def expand_catalogues(xm):
            return {
                k: v
                for cat in self.XMATCH_COLUMNS
                for k, v in xm.get(cat, {}).items()
            }

        def get_required_columns():
            return [
                col for cols in self.XMATCH_COLUMNS.values() for col in cols
            ]

        xmatches = [
            {"oid": xm["oid"]} | expand_catalogues(xm) for xm in xmatches
        ]
        xmatches = pd.DataFrame(
            xmatches, columns=["oid"] + get_required_columns()
        )
        xmatches = xmatches.rename(columns={"oid": "id"}).drop_duplicates(
            subset=["id"]
        )
        return xmatches[xmatches["id"].isin(self.detections.ids())].set_index(
            "id"
        )

    def clear_caches(self):
        """Clears the cache from detections and non-detections."""
        self.detections.clear_caches()
        self.non_detections.clear_caches()

    @decorators.logger
    def calculate_periods(self) -> pd.DataFrame:
        return self.detections.apply(
            extras.periods,
            fids=self.BANDS,
            kim=self.COMPUTE_KIM,
            n_harmonics=self.N_HARMONICS,
            factors=self.POWER_RATE_FACTORS,
        )

    @decorators.logger
    @decorators.columns_per_fid
    @decorators.fill_in_every_fid()
    def calculate_fats(self) -> pd.DataFrame:
        results = self.detections.apply(
            extras.turbofats, by_fid=True, features=self.FATS_FEATURES
        )
        return results.reset_index(
            "oid", drop=True
        )  # Superfluous index added by turbofats

    @decorators.logger
    @decorators.columns_per_fid
    @decorators.fill_in_every_fid()
    def calculate_mhps(self) -> pd.DataFrame:
        return self.detections.apply(
            extras.mhps, by_fid=True, t1=self.T1, t2=self.T2, flux=self.FLUX
        )

    @decorators.logger
    @decorators.columns_per_fid
    @decorators.fill_in_every_fid()
    def calculate_gp_drw(self) -> pd.DataFrame:
        return self.detections.apply(extras.gp_drw, by_fid=True)

    @decorators.logger
    @decorators.columns_per_fid
    @decorators.fill_in_every_fid()
    def calculate_iqr(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"iqr": self.detections.agg("mag_ml", stats.iqr, by_fid=True)}
        )

    def generate_features(
        self, exclude: set[str] | None = None
    ) -> pd.DataFrame:
        """Create a data frame with all required features.

        Args:
            exclude: Extractor(s) to exclude

        Returns:
            pd.DataFrame: Feature indexed by object with two-level columns (feature name and band)
        """
        if not self.detections.ids().size:
            self.logger.debug(
                "No objects present after filtering, skipping feature generation"
            )
            return pd.DataFrame()
        exclude = exclude or set()  # Empty default
        exclude |= self._AUTO_EXCLUDE  # Add all permanently excluded
        # Add prefix to exclude, unless already provided
        exclude = {
            name if name.startswith(self._PREFIX) else f"{self._PREFIX}{name}"
            for name in exclude
        }

        # Select all methods that start with prefix unless excluded
        methods = {
            name
            for name in dir(self)
            if name.startswith(self._PREFIX) and name not in exclude
        }

        # Compute all features and join into single dataframe
        return pd.concat(
            (getattr(self, method)() for method in methods),
            axis="columns",
            copy=False,
        )
