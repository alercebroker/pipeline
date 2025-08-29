import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any
from datetime import date, datetime


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
        
        # Build message for output
        output_message = {
            "oid": oid,
            "measurement_id": unique_measurement_ids_long,
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
        elif isinstance(obj, (date, datetime)):
            return obj.isoformat() 
        elif isinstance(obj, pd._libs.missing.NAType): 
            return None
        return super().default(obj)
