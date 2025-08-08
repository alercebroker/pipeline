import pandas as pd
import numpy as np
import json
from typing import Dict, List
from collections import defaultdict


def parse_output_correction(corrected_data: dict, measurement_ids: dict, oids: list) -> List[Dict]:
    """
    Function to parse the output of the correction multisurvey step regardless of survey of the data
    It groups by the oid all of the inner dataframes from corrected data and returns a list of dicts corresponding
    to the output format of the correction multisurvey step

    Args:
        corrected_data: Dictionary with the output of the correction multisurvey step
        measurement_ids: List of measurement IDs
        oids: List of object IDs
    
    Returns:
        Dictionary with the output of the correction multisurvey step
    """
    
    valid_oids = set(oids)
    
    # Pre-process measurement_ids
    processed_measurement_ids = {
        oid: [int(id_str) for id_str in measurement_ids[oid]] 
        if measurement_ids[oid] and isinstance(measurement_ids[oid][0], str)
        else measurement_ids[oid]
        for oid in oids
    }
    
    # Pre-allocate final structure
    grouped_data = {data_type: defaultdict(list) for data_type in corrected_data.keys()}
    
    # Process each of the corrected dataframes
    for data_type, df in corrected_data.items():
        if df.empty or "oid" not in df.columns:
            continue
        
        # Select only valid oids
        filtered_df = df[df["oid"].isin(valid_oids)]
        if filtered_df.empty:
            continue
        
        # Convert to records
        records = filtered_df.to_dict('records')
        
        # Replace the nan and infinity values with None
        for record in records:
            oid = record["oid"]
            
            cleaned_record = {}
            for key, value in record.items():
                try:
                    if pd.isna(value) or (isinstance(value, (int, float)) and (np.isinf(value) or np.isnan(value))):
                        cleaned_record[key] = None
                    else:
                        cleaned_record[key] = value
                except (TypeError, ValueError):
                    cleaned_record[key] = value
            
            grouped_data[data_type][oid].append(cleaned_record)
    
    # Convert defaultdict to regular dict
    # If an oid is not present in the corrected data, we add an empty list
    for data_type in grouped_data:
        grouped_data[data_type] = {oid: grouped_data[data_type].get(oid, []) for oid in oids}
    
    # Build the output where each result is a dict with oid, measurement_id and data from the corrected data cleaned
    # It works independently of the survey
    return [
        {
            "oid": oid,
            "measurement_id": processed_measurement_ids[oid],
            **{data_type: grouped_data[data_type][oid] for data_type in corrected_data.keys()}
        }
        for oid in oids
    ]


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
