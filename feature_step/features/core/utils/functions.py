import pandas as pd


def collapse_fid_columns(df: pd.DataFrame, mapping: dict = {}) -> pd.DataFrame:
    def join(lvls):
        name, fid = lvls
        return f"{name}_{mapping.get(fid, fid)}" if len(fid) == 1 else name

    df.columns = df.columns.map(join)
    return df
