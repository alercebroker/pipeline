import functools
from typing import Sequence

import numpy as np
import pandas as pd


def reformat(df: pd.DataFrame, fids: Sequence[str]) -> pd.Series:
    return df.reindex(fids).unstack()


def empty(indices: Sequence[str], fids: tuple[str, ...]) -> pd.Series:
    return pd.Series(np.nan, index=multiindex(tuple(indices), fids))


@functools.lru_cache()
def multiindex(indices: tuple[str, ...], fids: tuple[str, ...]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product([indices, fids], names=(None, "fid"))
