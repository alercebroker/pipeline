# Function that receives a list of dicts, creates a dict of a pd series for each of them, applying a schema file,
#  then returns a pandas dataframe

import pandas as pd
from typing import Dict, List, Any, Union


def apply_schema(data: list[dict], schema: dict) -> pd.DataFrame:
    """
    Function that receives a list of dicts, creates a dict of a pd series for each of them, applying a schema file,
    then returns a pandas dataframe after applying the schema.
    """
    return pd.DataFrame({k: pd.Series(v) for k, v in data.items()}).astype(schema)


def flatten_nested_data(data: List[Dict]) -> List[Dict]:
    """
    Flatten nested dictionaries by bringing extra_fields to top level. We do this to be able to apply the schema
    to all the fields in the data and returns the list of fields inside the extra fields so they can be re nested after
    applying the schema
    Parameters
    ----------
    data : List[Dict]
        List of dictionaries to be flattened
    Returns
    -------
    List[Dict]
        List of flattened dictionaries
    List[str]
        List of fields inside the extra fields

    """
    flattened = []
    all_extra_fields = set()
    for record in data:
        flat_record = {}

        # Copy all non-nested fields to mantain the structure
        for key, value in record.items():
            if key != "extra_fields":
                flat_record[key] = value

        # Flatten extra_fiels data one by one
        if "extra_fields" in record and record["extra_fields"]:
            for key, value in record["extra_fields"].items():
                flat_record[f"{key}"] = value
                all_extra_fields.add(key)

        flattened.append(flat_record)
    extra_fields_list = list(all_extra_fields)

    return flattened, extra_fields_list


def renest_extra_fields_schema(
    df: pd.DataFrame, extra_fields: List[str], schema: Dict[str, Any]
) -> pd.DataFrame:
    """
    Function that receives a dataframe with extra fields and a schema, and returns a dataframe
    with only the fields that are present in both extra_fields and schema nested into extra_fields column.
    All fields from extra_fields list are dropped from the dataframe columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with potential extra fields
    extra_fields : List[str]
        List of field names to consider for nesting
    schema : Dict[str, Any]
        Schema dictionary where keys are field names and values are dtypes

    Returns
    -------
    pd.DataFrame
        Dataframe with filtered extra fields nested and all extra_fields dropped
    """
    result_df = df.copy()

    schema_keys = set(schema.keys())
    fields_to_nest = [
        field for field in extra_fields if field in schema_keys and field in df.columns
    ]

    if len(fields_to_nest) > 0:
        result_df["extra_fields"] = result_df[fields_to_nest].to_dict("records")

    fields_to_drop = [field for field in extra_fields if field in df.columns]
    if fields_to_drop:
        result_df = result_df.drop(columns=fields_to_drop)

    return result_df


def apply_schema_flatten_data(data: list[dict], schema: dict) -> pd.DataFrame:
    """
    Function that receives a list of dicts, creates a dict of a pd series for each of them, applying a schema file,
    then returns a pandas dataframe after applying the schema. It first flattens the nested data, then applies the schema, and
    finally re-nests the extra fields before returning the dataframe
    Parameters
    ----------
    data : list[dict]
        List of dictionaries to be flattened
    schema : dict
        Schema to be applied
    Returns
    -------
    pd.DataFrame
    """

    flattened_data, extra_fields = flatten_nested_data(data)
    schematized_data = {
        col: pd.Series([item.get(col) for item in flattened_data], dtype=dtype)
        for col, dtype in schema.items()
    }
    dataframe_unnested = pd.DataFrame(schematized_data)
    renested_frame = renest_extra_fields_schema(dataframe_unnested, extra_fields, schema)
    return renested_frame
