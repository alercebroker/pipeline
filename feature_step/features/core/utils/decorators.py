import functools
from typing import Any

from .functions import fill_index


def logger(method):
    """Decorated method must produce a two level column data frame"""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        _, _, name = method.__name__.rpartition(self._PREFIX)
        self.logger.info(f"Computing {' '.join(name.upper().split('_'))}...")
        df = method(self, *args, **kwargs)
        self.logger.info(f"Done: {df.columns.size} feature(s) computed")

        details = df.columns.to_frame().groupby(level=0)["fid"].unique()
        for idx in details.index:
            self.logger.debug(
                f"  Feature {idx} for band(s): {details.loc[idx]}"
            )
        return df

    return wrapper


def columns_per_fid(method):
    """Decorated method must produce a multi-indexed data frame with `fid` as a named level"""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return method(self, *args, **kwargs).unstack("fid")

    return wrapper


def add_fid(fid: Any):
    """Adds a sub-column called `fid` with the given value to all columns"""

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            df = method(self, *args, **kwargs)
            return (
                df.assign(fid=fid).set_index("fid", append=True).unstack("fid")
            )

        return wrapper

    return decorator


def fill_in_every_fid(counters: str = None):
    """Decorated method must produce a multi-indexed data frame with two levels, `oid` and `fid` (in that order)"""

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            df = method(self, *args, **kwargs)
            if self.BANDS:
                return fill_index(df, counters=counters, fid=self.BANDS)
            return df

        return wrapper

    return decorator
