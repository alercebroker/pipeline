from typing import List


from ..extractors.dummy_extractor import DummyExtractor
from ..core.base import FeatureExtractorComposite, FeatureExtractor
from ..extractors.timespan_extractor import TimespanExtractor
from ..extractors.coordinate_extractor import CoordinateExtractor
from ..extractors.period_extractor import PeriodExtractor
from ..extractors.tde_extractor import TDETailExtractor
from ..extractors.color_feature_extractor import ColorFeatureExtractor



class LSSTFeatureExtractor(FeatureExtractorComposite):
    version = "1.0.1"

    def _instantiate_extractors(self) -> List[FeatureExtractor]:
        bands = ["u", "g", "r", "i", "z", "y"]
        #bands = range(6)  # LSST bands as integers 0-5

        feature_extractors = [
            ColorFeatureExtractor(bands, just_flux=False),
            TimespanExtractor(),
            CoordinateExtractor(),
            TDETailExtractor(bands),
            PeriodExtractor(
                bands,
                unit="magnitude",
                smallest_period=0.045,
                largest_period=50.0,
                trim_lightcurve_to_n_days=500.0,
                min_length=15,
                use_forced_photo=False,
                return_power_rates=True,
                shift=0.1,
            ),  # TODO: consider LPVs + be within comp. budget
        ]
        return feature_extractors
