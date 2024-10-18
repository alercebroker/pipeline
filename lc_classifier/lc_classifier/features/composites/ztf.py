from typing import List

from ..core.base import FeatureExtractorComposite, FeatureExtractor
from ..extractors.color_feature_extractor import ColorFeatureExtractor
from ..extractors.allwise_colors_feature_extractor import AllwiseColorsFeatureExtractor
from ..extractors.panstarrs_feature_extractor import PanStarrsFeatureExtractor
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
from ..extractors.tde_extractor import TDETailExtractor
from ..extractors.tde_extractor import FleetExtractor
from ..extractors.tde_extractor import ColorVariationExtractor
from ..extractors.ulens_extractor import MicroLensExtractor
from ..extractors.reference_feature_extractor import ReferenceFeatureExtractor


class ZTFFeatureExtractor(FeatureExtractorComposite):
    version = "1.0.1"

    def _instantiate_extractors(self) -> List[FeatureExtractor]:
        bands = list("gr")

        feature_extractors = [
            ColorFeatureExtractor(bands, just_flux=False),
            AllwiseColorsFeatureExtractor(bands),
            PanStarrsFeatureExtractor(),
            MHPSExtractor(bands, unit="diff_flux"),
            MHPSExtractor(bands, unit="diff_flux", t1=365.0, t2=30.0),
            GPDRWExtractor(bands, unit="diff_flux"),
            PeriodExtractor(
                bands,
                unit="magnitude",
                smallest_period=0.045,
                largest_period=100.0,
                trim_lightcurve_to_n_days=1000.0,
                min_length=15,
                use_forced_photo=True,
                return_power_rates=True,
                shift=0.1,
            ),  # TODO: consider LPVs + be within comp. budget
            FoldedKimExtractor(bands, unit="magnitude"),
            HarmonicsExtractor(bands, unit="magnitude", use_forced_photo=True),
            TurboFatsExtractor(bands, unit="magnitude"),
            SPMExtractor(
                bands,
                unit="diff_flux",
                redshift=None,
                extinction_color_excess=None,
                forced_phot_prelude=30.0,
            ),
            TDETailExtractor(bands),
            FleetExtractor(bands),
            ColorVariationExtractor(window_len=10, band_1="g", band_2="r"),
            SNExtractor(bands, unit="diff_flux", use_forced_photo=True),
            MicroLensExtractor(bands),
            ReferenceFeatureExtractor(bands),
            TimespanExtractor(),
            CoordinateExtractor(),
        ]
        return feature_extractors
