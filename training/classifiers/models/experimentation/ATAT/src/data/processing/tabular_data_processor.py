import numpy as np
import pandas as pd

from schema import ZTF_ff_columns_to_PROD

def processing_tabular_data(tabular, order_cols, col_name):
    columns_wo_oid = [c for c in tabular.columns if c != 'oid']
    assert set(columns_wo_oid).issubset(set(ZTF_ff_columns_to_PROD.keys()))
    tabular = tabular.rename(
        columns=ZTF_ff_columns_to_PROD)
    for col in tabular.columns:
        if col != 'oid':
            tabular[col] = pd.to_numeric(tabular[col], errors="coerce")
            tabular[col] = tabular[col].fillna(-np.inf)
    tabular[col_name] = tabular[order_cols].values.tolist()
    tabular = tabular[['oid', col_name]]
    return tabular

def processing_features(features, dict_info, oid, col_name):
    features = unify_tabular_data(oid, tabular=features)
    features = features.drop(dict_info["md_col_names"], axis="columns")
    features = processing_tabular_data(features, dict_info['feat_cols'], col_name)
    return features

def processing_metadata(metadata, dict_info, oid, col_name):
    metadata = unify_tabular_data(oid, tabular=metadata)
    metadata = metadata[["oid"] + dict_info["md_col_names"]]
    metadata = processing_tabular_data(metadata, dict_info['md_cols'], col_name)
    return metadata

def unify_tabular_data(oid, tabular):
    ao_tabular = tabular[~tabular["fid"].isin([None])]
    ao_aux_tabular = tabular[tabular["fid"].isin([None])]
    ao_tabular["name_fid"] = ao_tabular["name"] + "_" + ao_tabular["fid"]

    diccionario = {}
    for _, row in ao_tabular.iterrows():
        diccionario[row["name_fid"]] = [row["value"]]
        diccionario["oid"] = [oid]

    for _, row in ao_aux_tabular.iterrows():
        diccionario[row["name"]] = [row["value"]]

    return pd.DataFrame(diccionario)