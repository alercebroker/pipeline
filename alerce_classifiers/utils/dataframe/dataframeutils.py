import pandas as pd
import numpy as np

class DataframeUtils:
    @staticmethod
    def pad_list(lc: pd.DataFrame, nepochs: int, max_epochs: int) -> np.ndarray:
        pad_num = max_epochs - nepochs
        if pad_num >= 0:
            return np.pad(lc, (0, pad_num), "constant", constant_values=(0, 0))
        else:
            return np.array(lc)[np.linspace(0, nepochs - 1, num=max_epochs).astype(int)]
        
    @staticmethod
    def get_max_epochs(pd_output: pd.DataFrame) -> int:
        return pd_output.groupby(["aid", "BAND"]).count()["FLUXCAL"].max()
    
    @staticmethod
    def create_mask(lc: pd.DataFrame) -> np.ndarray:
        return (lc != 0).astype(float)
    
    @staticmethod
    def normalizing_time(time_fid: pd.Series) -> pd.Series:
        mask_min = 9999999999 * (time_fid == 0).astype(float)
        t_min = np.min(time_fid + mask_min)
        return (time_fid - t_min) * (~(time_fid == 0)).astype(float)
    
    @staticmethod
    def separate_by_filter(
        feat_time_series: np.ndarray, bands: np.ndarray, max_epochs: int
    ) -> np.ndarray:
        timef_array = np.array(feat_time_series)
        band_array = np.array(bands)
        colors = {"u": "b", "g": "g", "r": "r", "i": "orange", "z": "brown", "Y": "k"}
        final_array = []
        for i, color in enumerate(colors.keys()):
            aux = timef_array[band_array == color]
            nepochs = len(aux)
            final_array += [DataframeUtils.pad_list(aux, nepochs, max_epochs)]
        return np.stack(final_array, 1)