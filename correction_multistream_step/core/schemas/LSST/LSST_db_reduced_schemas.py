import pandas as pd


DType = (
    pd.Int32Dtype
    | pd.Int64Dtype
    | pd.Float32Dtype
    | pd.Float64Dtype
    | pd.BooleanDtype
    | pd.StringDtype
)

dia_non_detection_lsst_db = {
    "band": pd.Int32Dtype(),
    "diaNoise": pd.Float32Dtype(),
    "ccdVisitId": pd.Int64Dtype(),
    "oid": pd.Int64Dtype(),
    "mjd": pd.Float64Dtype(),
}

dia_forced_sources_lsst_db = {
    "detector": pd.Int32Dtype(),
    "oid": pd.Int64Dtype(),
    "psfFluxErr": pd.Float64Dtype(),
    "measurement_id": pd.Int64Dtype(),
    "visit": pd.Int64Dtype(),
    "psfFlux": pd.Float64Dtype(),
    "mjd": pd.Float64Dtype(),
    "dec": pd.Float64Dtype(),
    "ra": pd.Float64Dtype(),
    "band": pd.Int32Dtype(),
    "new": pd.BooleanDtype(),


}

dia_source_lsst_db = {
    "parentDiaSourceId": pd.Int64Dtype(),
    "psfFluxErr": pd.Float64Dtype(),
    "psfFlux_flag_edge": pd.BooleanDtype(),
    "measurement_id": pd.Int64Dtype(),
    "oid": pd.Int64Dtype(),
    "psfFlux": pd.Float64Dtype(),
    "psfFlux_flag": pd.BooleanDtype(),
    "psfFlux_flag_noGoodPixels": pd.BooleanDtype(),
    "raErr": pd.Float32Dtype(),
    "decErr": pd.Float32Dtype(),
    "ra": pd.Float64Dtype(),
    "band": pd.Int32Dtype(),
    "dec": pd.Float64Dtype(),
    "mjd": pd.Float64Dtype(),
    "new": pd.BooleanDtype(),
}