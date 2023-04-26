import functools
from typing import Any

import pandas as pd


def fill_index(df: pd.DataFrame | pd.Series, *, fill_value: Any = pd.NA, **kwargs):
    if not kwargs:
        raise ValueError("At least one additional index must be included to fill in")
    if "aid" in kwargs:
        raise ValueError("The index `aid` cannot be pre-specified")
    check = ["aid"] + list(kwargs.keys())
    if set(df.index.names) != set(check):
        raise ValueError(f"DataFrame has MultiIndex with {df.index.names}. Requested filling: {check}")
    aids = df.index.get_level_values("aid").unique()
    values = [aids] + [kwargs[k] for k in df.index.names if k != "aid"]
    return df.reindex(pd.MultiIndex.from_product(values, names=df.index.names), fill_value=fill_value)


def columns_per_fid(method):
    """Decorated method must produce a multi-indexed data frame with `fid` as a named level"""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        df = method(self, *args, **kwargs).unstack("fid")
        if self.BANDS_MAPPING:
            df.rename(columns=self.BANDS_MAPPING, level="fid", inplace=True)
        df.columns = df.columns.map(lambda lvls: f"{'_'.join(str(lvl) for lvl in lvls)}")
        return df

    return wrapper


def add_fid(fid: Any):
    """Adds a sub-column called `fid` with the given value to all columns"""

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            df = method(self, *args, **kwargs)
            return df.assign(fid=fid).set_index("fid", append=True).unstack("fid")

        return wrapper

    return decorator


def fill_in_every_fid(*, fill_value: Any = pd.NA):
    """Decorated method must produce a multi-indexed data frame with two levels, `aid` and `fid` (in that order)"""

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            df = method(self, *args, **kwargs)
            if self.BANDS:
                return fill_index(df, fill_value=fill_value, fid=self.BANDS)
            return df

        return wrapper

    return decorator
