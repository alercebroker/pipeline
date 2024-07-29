from typing import List

from ..core.base import FeatureExtractorComposite, FeatureExtractor
from ..extractors.color_feature_extractor import ColorFeatureExtractor
from ..extractors.period_extractor import PeriodExtractor
from ..extractors.folded_kim_extractor import FoldedKimExtractor
from ..extractors.harmonics_extractor import HarmonicsExtractor
from ..extractors.mhps_extractor import MHPSExtractor
from ..extractors.turbofats_extractor import TurboFatsExtractor
from ..extractors.spm_extractor import SPMExtractor
from ..extractors.sn_extractor import SNExtractor
from ..extractors.timespan_extractor import TimespanExtractor
from ..extractors.coordinate_extractor import CoordinateExtractor


class ElasticcFeatureExtractor(FeatureExtractorComposite):
    def _instantiate_extractors(self) -> List[FeatureExtractor]:
        bands = list("ugrizY")
        unit = "diff_flux"

        feature_extractors = [
            ColorFeatureExtractor(bands, just_flux=True),
            MHPSExtractor(bands, unit),
            PeriodExtractor(
                bands,
                unit,
                smallest_period=0.045,
                largest_period=50.0,
                trim_lightcurve_to_n_days=500.0,
                min_length=15,
                use_forced_photo=True,
                return_power_rates=True,
            ),
            FoldedKimExtractor(bands, unit),
            HarmonicsExtractor(bands, unit, use_forced_photo=True),
            TurboFatsExtractor(bands, unit),
            SPMExtractor(
                bands,
                unit,
                redshift="REDSHIFT_HELIO",
                extinction_color_excess="MWEBV",
                forced_phot_prelude=30.0,
            ),
            SNExtractor(bands, unit, use_forced_photo=True),
            TimespanExtractor(),
            CoordinateExtractor(),
        ]
        return feature_extractors
