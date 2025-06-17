from typing import Any

import pandas as pd

DType = (
    pd.Int32Dtype
    | pd.Int64Dtype
    | pd.Float32Dtype
    | pd.Float64Dtype
    | pd.BooleanDtype
    | pd.StringDtype
)

Message = dict[str, Any]
