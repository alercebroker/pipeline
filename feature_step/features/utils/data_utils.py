import pandas as pd
import os
import uuid
from typing import List


def clean_and_flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana columnas con listas y remueve saltos de línea en strings."""
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df = df.explode(col)
        df[col] = df[col].apply(lambda x: str(x).replace('\n', ' ') if isinstance(x, str) else x)
    return df


def save_astro_objects_to_csvs(
    astro_objects: List["AstroObject"],
    messages_to_process: List[dict],
    base_folder: str = "csvs",
) -> str:
    """Guarda detections y features de cada AstroObject en CSVs por OID.

    Crea una carpeta base si no existe y un subfolder por batch con UUID.
    Retorna la ruta del folder del batch.
    """
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    batch_id = str(uuid.uuid4())
    batch_folder = os.path.join(base_folder, batch_id)
    os.makedirs(batch_folder)
    #print(len(astro_objects))

    for i, (ao, msg) in enumerate(zip(astro_objects, messages_to_process)):
        oid = getattr(ao, "oid", msg.get("oid", f"obj_{i}"))
        #print(oid)
        detections_csv_path = os.path.join(batch_folder, f"{oid}_detections.csv")
        features_csv_path = os.path.join(batch_folder, f"{oid}_features.csv")

        detections_df = clean_and_flatten_columns(ao.detections)
        features_df = clean_and_flatten_columns(ao.features)

        if "sid" in detections_df.columns:
            detections_df = detections_df.drop(columns=["sid"])
        if "sid" in features_df.columns:
            features_df = features_df.drop(columns=["sid"])

        detections_df.to_csv(detections_csv_path, index=False)
        features_df.to_csv(features_csv_path, index=False)

        print(f"Saved: {detections_csv_path}")
        print(f"Saved: {features_csv_path}")

    return batch_folder