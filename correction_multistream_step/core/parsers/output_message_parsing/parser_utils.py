import pandas as pd
import numpy as np
import json


def parse_output_correction(corrected_data: dict, measurement_ids: list, oids: list) -> pd.DataFrame:
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
    def _create_grouped_data(df: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
        """Function to safely create grouped data even when the DataFrame is empty"""
        if not df.empty and "oid" in df.columns:
            return df.groupby("oid")
        else:
            return pd.DataFrame(columns=["oid"]).groupby("oid")
    
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Function to clean DataFrame by replacing NaN and inf values"""
        return df.replace({np.nan: None, pd.NA: None, -np.inf: None})
    
    def _get_group_safely(grouped_data, oid: str, fallback_value: list = None) -> list:
        """Function to safely get group data with a default value when there is no data for the oid"""
        try:
            group_df = grouped_data.get_group(oid)
            cleaned_df = _clean_dataframe(group_df)
            return cleaned_df.to_dict("records")
        except KeyError:
            return fallback_value or []
    
    # For each key in the corrected data dictionary
    # Create the group by oid without getting an error if the DataFrame is empty
    grouped_data = {}
    for data_type, df in corrected_data.items():
        grouped_data[data_type] = _create_grouped_data(df)
    
    # Create a list to store the output dicts
    output = []
    
    # Group by for each oid in the list
    for oid in oids:        
        # Get measurement IDs for this OID
        unique_measurement_ids = measurement_ids[oid]
        unique_measurement_ids_long = [int(id_str) for id_str in unique_measurement_ids]
        
        # Build the output message for this OID
        # First we create the base message and then we add the grouped data for each of the keys
        output_message = {
            "oid": oid,
            "measurement_id": unique_measurement_ids_long,
        }
        
        # Add all data types from corrected_data using original keys
        for data_type, grouped in grouped_data.items():
            output_message[data_type] = _get_group_safely(grouped, oid)
        
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
        return super().default(obj)
