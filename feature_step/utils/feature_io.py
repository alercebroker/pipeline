import os
from typing import List
import pandas as pd


def find_feature_files(base_folder: str) -> List[str]:
    """Busca recursivamente CSVs de features que terminen en '_features.csv' o 'features.csv'."""
    feature_files: List[str] = []
    for root, _, files in os.walk(base_folder):
        for fname in files:
            lower = fname.lower()
            if lower.endswith("_features.csv") or lower.endswith("features.csv"):
                feature_files.append(os.path.join(root, fname))
    return feature_files


def _append_valid_suffix(base: pd.Series, maybe_suffix: pd.Series) -> pd.Series:
    s = maybe_suffix.astype(str)
    s = s.where(~s.isin(["nan", "NaN", "None", "", "<NA>"]), "")
    return base + s.apply(lambda v: f"_{v}" if v else "")


def load_features_series(fpath: str) -> pd.Series:
    """Carga un CSV de features en formato Series con índice=nombre de feature y valor=valor.

    Soporta:
    - Formato largo: columnas ('name','value') o ('feature','value').
    - Formato ancho: una fila con columnas=features.
    Si existen columnas 'fid' o 'band', se anexan al nombre solo si hay valor válido.
    En caso de duplicados, se toma el primer valor.
    """
    df = pd.read_csv(fpath)

    # Formato largo con 'name' y 'value'
    if {"name", "value"}.issubset(df.columns):
        name_col = df["name"].astype(str)
        if "band" in df.columns:
            name_col = _append_valid_suffix(name_col, df["band"])  # añade _<band> si existe
        elif "fid" in df.columns:
            name_col = _append_valid_suffix(name_col, df["fid"])   # añade _<fid> si existe
        tmp = pd.DataFrame({"feature": name_col, "value": df["value"]})
        tmp = tmp.dropna(subset=["feature"]).drop_duplicates(subset=["feature"], keep="first")
        return tmp.set_index("feature")["value"]

    # Formato largo alternativo con 'feature' y 'value'
    if {"feature", "value"}.issubset(df.columns):
        name_col = df["feature"].astype(str)
        if "band" in df.columns:
            name_col = _append_valid_suffix(name_col, df["band"])  # añade _<band> si existe
        elif "fid" in df.columns:
            name_col = _append_valid_suffix(name_col, df["fid"])   # añade _<fid> si existe
        tmp = pd.DataFrame({"feature": name_col, "value": df["value"]})
        tmp = tmp.dropna(subset=["feature"]).drop_duplicates(subset=["feature"], keep="first")
        return tmp.set_index("feature")["value"]

    # Fallback: formato ancho (una fila con columnas = features)
    if len(df) >= 1:
        series = df.iloc[0]
        for drop_col in [
            "oid",
            "sid",
            "index",
            "Unnamed: 0",
            "objectId",
            "objectid",
            "fid",
            "band",
        ]:
            if drop_col in series.index:
                series = series.drop(labels=[drop_col])
        series = series[~series.index.duplicated(keep="first")]
        return series

    return pd.Series(dtype=float)
