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
from ..extractors.gp_drw_extractor import GPDRWExtractor


class ZTFFeatureExtractor(FeatureExtractorComposite):
    def _instantiate_extractors(self) -> List[FeatureExtractor]:
        bands = list('gr')

        feature_extractors = [
            ColorFeatureExtractor(bands, unit='magnitude'),
            MHPSExtractor(bands, unit='diff_flux'),
            GPDRWExtractor(bands, unit='diff_flux'),

            PeriodExtractor(
                bands,
                unit='magnitude',
                smallest_period=0.045,
                largest_period=100.0,
                trim_lightcurve_to_n_days=1000.0,
                min_length=15,
                use_forced_photo=True,
                return_power_rates=True,
                shift=0.1
            ),  # TODO: consider LPVs + be within comp. budget
            FoldedKimExtractor(bands, unit='magnitude'),
            HarmonicsExtractor(bands, unit='magnitude', use_forced_photo=True),
            TurboFatsExtractor(bands, unit='diff_flux'),
            SPMExtractor(
                bands, unit='diff_flux',
                redshift=None,
                extinction_color_excess=None,
                forced_phot_prelude=30.0
            ),
            SNExtractor(
                bands, unit='diff_flux', use_forced_photo=True
            ),
            TimespanExtractor(),
            CoordinateExtractor()
        ]
        return feature_extractors
