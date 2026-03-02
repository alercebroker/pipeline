# Function that receives a list of dicts, creates a dict of a pd series for each of them, applying a schema file,
#  then returns a pandas dataframe

import pandas as pd

def apply_schema(data: list[dict], schema: dict) -> pd.DataFrame:
    """
    Function that receives a list of dicts, creates a dict of a pd series for each of them, applying a schema file,
    then returns a pandas dataframe after applying the schema. 
    """
    if not data:
        return pd.DataFrame(columns=list(schema.keys()))
    columns_dict = {}
    for row in data:
        for key, value in row.items():
            if key not in columns_dict:
                columns_dict[key] = []
            columns_dict[key].append(value)
    
    return pd.DataFrame({k: pd.Series(v) for k, v in columns_dict.items()}).astype(schema)
