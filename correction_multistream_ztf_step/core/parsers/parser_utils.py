import pandas as pd
import numpy as np
import json


def get_fid(fid_as_int: int):
    fid = {1: "g", 2: "r", 0: None, 12: "gr", 3: "i"}
    try:
        return fid[fid_as_int]
    except KeyError:
        return fid_as_int


def parse_output(result: dict):
    result["detections"] = pd.DataFrame(result["detections"]).groupby("oid")

    try:  # If it has at least one non-detection
        result["non_detections"] = pd.DataFrame(result["non_detections"]).groupby("oid")
    except KeyError:  # And if there's not a single ndet for oid
        result["non_detections"] = pd.DataFrame(columns=["oid"]).groupby("oid")
    output = []

    for oid, dets in result["detections"]:

        dets = dets.replace(
            {np.nan: None, pd.NA: None, -np.inf: None}
        )  # Avoid NaN in the final results or infinite
        for field in [
            "e_ra",
            "e_dec",
        ]:  # Replace the e_ra/e_dec converted to None back to float nan per avsc formatting
            dets[field] = dets[field].apply(lambda x: x if pd.notna(x) else float("nan"))
        unique_measurement_ids = result["measurement_ids"][oid]
        unique_measurement_ids_long = [int(id_str) for id_str in unique_measurement_ids]

        detections_result = dets.to_dict("records")

        output_message = {
            "oid": oid,
            "measurement_id": unique_measurement_ids_long,
            "meanra": result["coords"][oid]["meanra"],
            "meandec": result["coords"][oid]["meandec"],
            "detections": detections_result,
        }

        try:
            output_message["non_detections"] = (
                result["non_detections"].get_group(oid).to_dict("records")
            )
        except KeyError:
            output_message["non_detections"] = []
        output.append(output_message)
    return output


def parse_output_df(
    detections_df: pd.DataFrame,
    non_detections_df: pd.DataFrame,
    coords_df: pd.DataFrame,
    measurement_ids: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parse detection data directly from DataFrames

    Args:
        detections_df: DataFrame with detection data
        non_detections_df: DataFrame with non-detection data
        coords: Coordinates data (dict or DataFrame)
        measurement_ids: Measurement IDs data
    """
    # Group directly - no intermediate conversions
    detections_grouped = detections_df.groupby("oid")

    # Handle non_detections grouping
    if not non_detections_df.empty:
        non_detections_grouped = non_detections_df.groupby("oid")
    else:
        non_detections_grouped = pd.DataFrame(columns=["oid"]).groupby("oid")

    output = []

    for oid, dets in detections_grouped:
        dets_clean = dets.copy()

        # Handle NaN/inf values
        dets_clean = dets_clean.replace({np.nan: None, pd.NA: None, -np.inf: None})

        # Get measurement IDs for this OID
        unique_measurement_ids = measurement_ids[oid]
        unique_measurement_ids_long = [int(id_str) for id_str in unique_measurement_ids]

        # Convert to records only once at the end
        detections_result = dets_clean.to_dict("records")

        # Get coordinates for this OID
        coord_row = (
            coords_df[coords_df["oid"] == oid].iloc[0]
            if not coords_df[coords_df["oid"] == oid].empty
            else {"meanra": None, "meandec": None}
        )

        output_message = {
            "oid": oid,
            "measurement_id": unique_measurement_ids_long,
            "meanra": coord_row["meanra"],
            "meandec": coord_row["meandec"],
            "detections": detections_result,
        }

        # Handle non_detections
        try:
            non_dets = non_detections_grouped.get_group(oid)
            # Apply same cleaning to non_detections if needed
            non_dets_clean = non_dets.replace({np.nan: None, pd.NA: None, -np.inf: None})
            output_message["non_detections"] = non_dets_clean.to_dict("records")
        except KeyError:
            output_message["non_detections"] = []

        output.append(output_message)

    return output


def parse_data_for_avro(obj):
    """Recursively clean data for Avro serialization"""
    if isinstance(obj, dict):
        return {key: parse_data_for_avro(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [parse_data_for_avro(item) for item in obj]
    elif pd.isna(obj) or isinstance(obj, pd._libs.missing.NAType):
        return None  # Avro understands null
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


class NumpyEncoder(json.JSONEncoder):
    """
    Encode data to formats able to be parsed via json dumps for the scribe multisurvey
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, pd._libs.missing.NAType):
            return None
        return super().default(obj)
