import abc
from typing import Any

import pandas as pd
from astropy.coordinates import SkyCoord
from scipy import stats

from .handlers import DetectionsHandler, NonDetectionsHandler
from .utils import decorators, extras


class BaseFeatureExtractor(abc.ABC):
    _PREFIX: str = "calculate_"
    SURVEYS: tuple[str, ...] = ()
    BANDS: tuple[str, ...] = ()
    BANDS_MAPPING: dict[str, Any] = {}
    EXTRA_COLUMNS: list[str] = []
    XMATCH_COLUMNS: list[str] = []
    USE_CORRECTED: bool = False
    FLUX: bool = False
    COMPUTE_KIM: bool = True
    N_HARMONICS: int = 7
    POWER_RATE_FACTORS: tuple[str, ...] = ("1/4", "1/3", "1/2", "2", "3", "4")
    MIN_DETECTIONS: int = 0
    MIN_DETECTIONS_IN_FID: int = 5
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
        non_detections: list[dict] | pd.DataFrame,
        xmatches: list[dict] | pd.DataFrame = None,
        *,
        legacy: bool = False
    ):
        common = dict(surveys=self.SURVEYS, bands=self.BANDS, legacy=legacy)

        if isinstance(detections, pd.DataFrame):
            detections = detections.reset_index().to_dict("records")
        self.detections = DetectionsHandler(detections, extras=self.EXTRA_COLUMNS, corr=self.USE_CORRECTED, **common)
        self._discard_detections()

        if isinstance(non_detections, pd.DataFrame):
            non_detections = non_detections.reset_index().to_dict("records")
        first_mjd = self.detections.get_aggregate("mjd", "min", by_fid=True)
        self.non_detections = NonDetectionsHandler(non_detections, first_mjd=first_mjd, **common)
        self.non_detections.match(self.detections)

        if isinstance(xmatches, pd.DataFrame):
            xmatches = xmatches.reset_index().to_dict("records")
        self.xmatches = self._create_xmatches(xmatches, legacy)
        self._periods = None

    @abc.abstractmethod
    def _discard_detections(self):
        self.detections.remove_objects_without_enough_detections(self.MIN_DETECTIONS)
        self.detections.remove_objects_without_enough_detections(self.MIN_DETECTIONS_IN_FID, by_fid=True)

    def _create_xmatches(self, xmatches: list[dict], legacy: bool) -> pd.DataFrame:
        id_col = "oid" if legacy else "aid"
        xmatches = pd.DataFrame(xmatches) if xmatches else pd.DataFrame(columns=[id_col] + self.XMATCH_COLUMNS)
        xmatches = xmatches.rename(columns={id_col: "id"})
        xmatches = xmatches[xmatches["id"].isin(self.detections.get_objects())]
        return xmatches.set_index("id")[self.XMATCH_COLUMNS]

    def clear_caches(self):
        self.detections.clear_caches()
        self.non_detections.clear_caches()

    @decorators.add_fid(0)
    def calculate_galactic_coordinates(self) -> pd.DataFrame:
        ra = self.detections.get_aggregate("ra", "mean")
        dec = self.detections.get_aggregate("dec", "mean")
        galactic = SkyCoord(ra, dec, frame="icrs", unit="deg").galactic
        # By construction, ra and dec indices should be the same
        return pd.DataFrame({"gal_b": galactic.b.degree, "gal_l": galactic.l.degree}, index=ra.index)

    def calculate_periods(self) -> pd.DataFrame:
        df = self.detections.apply_grouped(
            extras.periods,
            fids=self.BANDS,
            kim=self.COMPUTE_KIM,
            n_harmonics=self.N_HARMONICS,
            factors=self.POWER_RATE_FACTORS,
        )
        return df.rename(columns=self.BANDS_MAPPING, level="fid")

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid()
    def calculate_fats(self) -> pd.DataFrame:
        return self.detections.apply_grouped(extras.turbofats, by_fid=True, features=self.FATS_FEATURES)

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid()
    def calculate_mhps(self) -> pd.DataFrame:
        return self.detections.apply_grouped(extras.mhps, by_fid=True, t1=100, t2=10, flux=self.FLUX)

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid()
    def calculate_iqr(self) -> pd.DataFrame:
        return pd.DataFrame({"iqr": self.detections.get_aggregate("mag_ml", stats.iqr, by_fid=True)})

    def generate_features(self, exclude: set[str] | None = None) -> pd.DataFrame:
        exclude = exclude or set()  # Empty default
        # Add prefix to exclude, unless already provided
        exclude = {name if name.startswith(self._PREFIX) else f"{self._PREFIX}{name}" for name in exclude}

        # Select all methods that start with prefix unless excluded
        methods = {name for name in dir(self) if name.startswith(self._PREFIX) and name not in exclude}

        # Compute all features and join into single dataframe
        return pd.concat((getattr(self, method)() for method in methods), axis="columns", copy=False)
