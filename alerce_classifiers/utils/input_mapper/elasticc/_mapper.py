import numpy as np
import pandas as pd
from alerce_classifiers.base.dto import InputDTO

from alerce_classifiers.base.mapper import Mapper
from .dict_transform import FEAT_DICT

class ElasticcMapper(Mapper):
    _fid_mapper = {
        0: "u",
        1: "g",
        2: "r",
        3: "i",
        4: "z",
        5: "Y",
    }
    _rename_cols = {
        "mag": "FLUXCAL",
        "e_mag": "FLUXCALERR",
        "fid": "BAND",
        "mjd": "MJD",
    }
    _feat_dict = FEAT_DICT

    def _get_detections(self, lc: pd.DataFrame):
        pass

    def preprocess(input: InputDTO):
        return super().preprocess()
    
    def postprocess():
        pass