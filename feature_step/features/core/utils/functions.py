import pandas as pd


def fill_index(
    df: pd.DataFrame | pd.Series, *, counters: str = None, **kwargs
) -> pd.DataFrame | pd.Series:
    if not kwargs:
        raise ValueError(
            "At least one additional index must be included to fill in"
        )
    if "id" in kwargs:
        raise ValueError("The index `id` cannot be pre-specified")
    check = ["id"] + list(kwargs.keys())
    if set(df.index.names) != set(check):
        raise ValueError(
            f"DataFrame has MultiIndex with {df.index.names}. Requested filling: {check}"
        )
    oids = df.index.get_level_values("id").unique()
    values = [oids] + [kwargs[k] for k in df.index.names if k != "id"]

    df = df.reindex(pd.MultiIndex.from_product(values, names=df.index.names))
    if counters is None:
        return df

    cols = [c for c in df.columns if c.startswith(counters)]
    df[cols] = df[cols].fillna(0).astype(int)
    return df


def collapse_fid_columns(df: pd.DataFrame, mapping: dict = {}) -> pd.DataFrame:
    def join(lvls):
        name, fid = lvls
        return f"{name}_{mapping.get(fid, fid)}" if len(fid) == 1 else name

    df.columns = df.columns.map(join)
    return df
