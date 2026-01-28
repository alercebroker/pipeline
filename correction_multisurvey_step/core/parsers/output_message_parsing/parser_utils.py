import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any
from datetime import date, datetime
from collections import defaultdict


def get_measurement_ids(corrected_data: dict, survey: str) -> Dict[int, List[str]]:
    """
    Function to get the measurement IDs grouped by oid depending on the survey
    """

    result = defaultdict(list)

    if survey == "lsst":
        keys = ["sources", "ss_sources"]

    elif survey == "ztf":
        keys = ["detections", "previous_detections"]

    else:
        raise ValueError(f"Unknown survey: {survey}")

    for key in keys:
        df = corrected_data.get(key)

        if df is None or df.empty:
            continue

        # Make sure required columns exist
        if not {"oid", "measurement_id"}.issubset(df.columns):
            continue

        grouped = df.groupby("oid")["measurement_id"].apply(list)

        for oid, mids in grouped.items():
            result[oid].extend(mids)

    return dict(result)

def get_sids_from_messages(messages: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    """
    Function to get the sids for each oid in the messages

    Args:
        messages (List[Dict[str, Any]]): List of messages

    Returns:
        Dict[int, List[str]]: Dictionary with oids as keys and a single sid per oid as values
    """
    result = {}
    for message in messages:
        oid = message.get("oid")
        sid = message.get("sid")
        result[oid] = sid
    return result

def parse_output_correction(corrected_data: dict, measurement_ids: dict, oids: list, sids) -> List[Dict]:
    """
    Function to parse the output of the correction multisurvey step regardless of survey of the data
    It groups by the oid all of the inner dataframes from corrected data and returns a list of dicts corresponding
    to the output format of the correction multisurvey step

    Args:
        corrected_data: Dictionary with the output of the correction multisurvey step
        measurement_ids: List of measurement IDs
        oids: List of object IDs
        sids: Dict of survey IDs to object IDs
    
    Returns:
        Dictionary with the output of the correction multisurvey step
    """
    
    def _clean_dataframe_vectorized(df: pd.DataFrame) -> pd.DataFrame:
        """ Function to clean the dataframe from inf and NaN values in a vectorized way instead of the original code that iterated over the rows """
        if df.empty:
            return df
        
        # Replace inf and NaN values in one operation
        df = df.replace([np.inf, -np.inf, np.nan], None)
        return df
    
    grouped_data = {}
    valid_oids_set = set(oids)
    
    for data_type, df in corrected_data.items():
        if df.empty or "oid" not in df.columns:
            grouped_data[data_type] = {oid: [] for oid in oids}
        else:
            filtered_df = df[df["oid"].isin(valid_oids_set)]
            
            if filtered_df.empty:
                grouped_data[data_type] = {oid: [] for oid in oids}
            else:
                cleaned_df = _clean_dataframe_vectorized(filtered_df)
                
                grouped = cleaned_df.groupby("oid", sort=False)
                grouped_dict = {oid: group.to_dict("records") 
                              for oid, group in grouped}
                
                grouped_data[data_type] = {oid: grouped_dict.get(oid, []) 
                                         for oid in oids}
    
    # Create output list
    output = []
    for oid in oids:
        unique_measurement_ids = measurement_ids[oid]
        if isinstance(unique_measurement_ids[0], str):
            unique_measurement_ids_long = [int(id_str) for id_str in unique_measurement_ids]
        else:
            unique_measurement_ids_long = unique_measurement_ids
        
        sid = sids[oid]
        # Build message for output
        output_message = {
            "oid": oid,
            "measurement_id": unique_measurement_ids_long,
            "sid": sid
        }
        
        for data_type in corrected_data.keys():
            output_message[data_type] = grouped_data[data_type][oid]
        
        output.append(output_message)
    
    return output

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
        elif isinstance(obj, (date, datetime)):
            return obj.isoformat()  # Convert date/datetime to ISO string format
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()  # Handle pandas Timestamp objects
        return super().default(obj)

