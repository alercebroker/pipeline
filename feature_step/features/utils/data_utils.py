import pandas as pd
import os
import uuid
import hashlib
from typing import List
from lc_classifier.features.core.base import AstroObject


def clean_and_flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana columnas con listas y remueve saltos de línea en strings."""
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df = df.explode(col)
        df[col] = df[col].apply(lambda x: str(x).replace('\n', ' ') if isinstance(x, str) else x)
    return df


def save_astro_objects_to_csvs(
    astro_objects: List[AstroObject],
    messages_to_process: List[dict],
    base_folder: str = "csvs",
) -> str:
    """Guarda detections y features de cada AstroObject en CSVs por OID.

    Crea una carpeta base si no existe, organiza por class_name y luego por batch con UUID.
    Retorna la ruta del folder del batch.
    """
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # Obtener class_name del primer mensaje (asumiendo que el batch es de un solo objeto/clase)
    first_class_name = messages_to_process[0].get("class_name", "unknown") if messages_to_process else "unknown"
    
    # Crear estructura: base_folder/class_name/batch_id/
    class_folder = os.path.join(base_folder, first_class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    
    batch_id = str(uuid.uuid4())
    batch_folder = os.path.join(class_folder, batch_id)
    os.makedirs(batch_folder)
    #print(len(astro_objects))

    for i, (ao, msg) in enumerate(zip(astro_objects, messages_to_process)):
        oid = getattr(ao, "oid", msg.get("oid", f"obj_{i}"))
        
        # Generar un hash random corto para diferenciar archivos del mismo OID
        random_hash = hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]
        
        #print(oid)
        detections_csv_path = os.path.join(batch_folder, f"{oid}_{random_hash}_detections.csv")
        features_csv_path = os.path.join(batch_folder, f"{oid}_{random_hash}_features.csv")
        metadata_csv_path = os.path.join(batch_folder, f"{oid}_{random_hash}_metadata.csv")


        detections = ao.detections
        detections = detections[detections.unit == 'magnitude']
        detections_df = clean_and_flatten_columns(detections)
        features_df = clean_and_flatten_columns(ao.features)

        if "sid" in detections_df.columns:
            detections_df = detections_df.drop(columns=["sid"])
        if "sid" in features_df.columns:
            features_df = features_df.drop(columns=["sid"])

        detections_df.to_csv(detections_csv_path, index=False)
        features_df.to_csv(features_csv_path, index=False)

        # Save metadata (period, class_name, etc.)
        metadata = {
            'oid': [oid],
            'class_name': [msg.get('class_name', None)],
            'period': [msg.get('period', None)],
            'num_detections': [len(detections_df)]
        }
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(metadata_csv_path, index=False)

        print(f"Saved: {detections_csv_path}")
        print(f"Saved: {features_csv_path}")
        print(f"Saved: {metadata_csv_path}")

    return batch_folder