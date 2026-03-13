import os
import numpy as np
import pandas as pd
from typing import List
from utils.astro_objects import AstroObject, astro_object_from_dict


def query_ao_table(table: pd.DataFrame, name: str, check_unique: bool = True):
    ans_df = table[table["name"] == name]
    if check_unique and len(ans_df) > 1:
        raise Exception(f"Field {name} appears {len(ans_df)} times.")
    return ans_df["value"].values[0] if check_unique else ans_df


def get_shorten(filename: str):
    return filename.split("/")[-1].split("_")[0]


def all_features_from_astro_objects(astro_objects: List[AstroObject]) -> pd.DataFrame:
    first_object = astro_objects[0]
    features = first_object.features.drop_duplicates(subset=["name", "fid"]).set_index(["name", "fid"])
    indexes = features.index.values

    feature_list = []
    oids = []

    for astro_object in astro_objects:
        current_features = astro_object.features.drop_duplicates(subset=["name", "fid"]).set_index(["name", "fid"])

        try:
            values = current_features.loc[indexes]["value"].values
            feature_list.append(values)
        except KeyError:
            print(f'OID {astro_object.detections.oid.iloc[0]} no tiene todas las features necesarias')
            continue

        oid = query_ao_table(astro_object.metadata, "oid")
        oids.append(oid)

    df = pd.DataFrame(
        data=np.stack(feature_list, axis=0),
        index=oids,
        columns=["_".join([str(i) for i in pair]) for pair in indexes],
    )
    df.index.name = "oid"
    return df


def process_batch(batch_filename: str) -> pd.DataFrame:
    """Procesa un archivo de lote de objetos astronómicos y extrae sus features."""
    shorten = get_shorten(batch_filename)

    astro_objects_batch = pd.read_pickle(batch_filename)
    astro_objects_batch = [astro_object_from_dict(ao) for ao in astro_objects_batch]

    features_batch = all_features_from_astro_objects(astro_objects_batch)
    features_batch["shorten"] = shorten

    # Reemplazar '_nan' en nombres de columnas si existe
    updated_columns = [col.replace('_nan', '') if '_nan' in col else col for col in features_batch.columns]
    features_batch.columns = updated_columns

    return features_batch
