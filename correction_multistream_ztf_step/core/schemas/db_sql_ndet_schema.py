import pandas as pd

db_sql_non_detection_schema = {
    "band": pd.Int64Dtype(),
    "oid": pd.Int64Dtype(),
    "mjd": pd.Float64Dtype(),
    "diffmaglim": pd.Float64Dtype(),
    "sid": pd.Int32Dtype(),
    "tid": pd.Int32Dtype(),
}
