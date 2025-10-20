from typing import List


from ..extractors.dummy_extractor import DummyExtractor
from ..core.base import FeatureExtractorComposite, FeatureExtractor
from ..extractors.timespan_extractor import TimespanExtractor
from ..extractors.coordinate_extractor import CoordinateExtractor
from ..extractors.period_extractor import PeriodExtractor
from ..extractors.tde_extractor import TDETailExtractor
from ..extractors.color_feature_extractor import ColorFeatureExtractor
from ..extractors.mhps_extractor import MHPSExtractor
from ..extractors.gp_drw_extractor import GPDRWExtractor
from ..extractors.folded_kim_extractor import FoldedKimExtractor
from ..extractors.harmonics_extractor import HarmonicsExtractor
from ..extractors.turbofats_extractor import TurboFatsExtractor
from ..extractors.spm_extractor import SPMExtractor
from ..extractors.sn_extractor import SNExtractor
from ..extractors.ulens_extractor import MicroLensExtractor
from ..extractors.tde_extractor import FleetExtractor
from ..extractors.tde_extractor import ColorVariationExtractor



class LSSTFeatureExtractor(FeatureExtractorComposite):
    version = "1.0.1"

    def _instantiate_extractors(self) -> List[FeatureExtractor]:
        bands = ["u", "g", "r", "i", "z", "y"]
        #bands = range(6)  # LSST bands as integers 0-5

        feature_extractors = [
            ColorFeatureExtractor(bands, just_flux=False), #resta siempre es u-g,g-r,r-i ... , que pares usar?
            TimespanExtractor(), #calcula
            CoordinateExtractor(), #calcula
            TDETailExtractor(bands), #calcula
            PeriodExtractor( #calcula
                bands,
                unit="magnitude",
                smallest_period=0.045,
                largest_period=50.0,
                trim_lightcurve_to_n_days=500.0,
                min_length=3,
                use_forced_photo=False,
                return_power_rates=True,
                shift=0.1,
            ),  # TODO: consider LPVs + be within comp. budget

            MHPSExtractor(bands, unit="diff_flux"), #calcula
            MHPSExtractor(bands, unit="diff_flux", t1=365.0, t2=30.0), #calcula
            GPDRWExtractor(bands, unit="diff_flux"), #calcula

            FoldedKimExtractor(bands, unit="magnitude"), #calcula
            HarmonicsExtractor(bands, unit="magnitude", use_forced_photo=False), #calcula
            TurboFatsExtractor(bands, unit="magnitude"),  #calcula
            SPMExtractor( #calcula 
                bands,
                unit="diff_flux",
                redshift=None,
                extinction_color_excess=None,
                forced_phot_prelude=30.0,
            ),
            FleetExtractor(bands), #calcula
            ColorVariationExtractor(window_len=10, band_1="u", band_2="g"), # que pares usar?
            ColorVariationExtractor(window_len=10, band_1="g", band_2="r"), # que pares usar?
            ColorVariationExtractor(window_len=10, band_1="r", band_2="i"), # que pares usar?
            ColorVariationExtractor(window_len=10, band_1="i", band_2="z"), # que pares usar?
            ColorVariationExtractor(window_len=10, band_1="z", band_2="y"), # que pares usar?

            SNExtractor(bands, unit="diff_flux", use_forced_photo=False), #esto es photmetria forzada
            MicroLensExtractor(bands), #calcula, cambie treshold en e_brightness
        ]
        return feature_extractors
