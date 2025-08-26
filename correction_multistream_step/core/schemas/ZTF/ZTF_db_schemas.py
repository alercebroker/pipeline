import pandas as pd


DType = (
    pd.Int32Dtype
    | pd.Int64Dtype
    | pd.Float32Dtype
    | pd.Float64Dtype
    | pd.BooleanDtype
    | pd.StringDtype
)

non_detections_ztf_db = {
    "oid": pd.Int64Dtype(),
    "sid": pd.Int32Dtype(),
    "band": pd.Int32Dtype(),
    "mjd": pd.Float64Dtype(),
    "diffmaglim": pd.Float32Dtype(),
}

detections_ztf_db = {
    "oid": pd.Int64Dtype(),
    "sid": pd.Int32Dtype(),
    "tid": pd.Int32Dtype(),
    "pid": pd.Int64Dtype(),
    "band": pd.Int32Dtype(),
    "measurement_id": pd.Int64Dtype(),
    "mjd": pd.Float64Dtype(),
    "ra": pd.Float64Dtype(),
    "dec": pd.Float64Dtype(),
    "mag": pd.Float32Dtype(),
    "e_mag": pd.Float32Dtype(),
    "isdiffpos": pd.Int32Dtype(),
    "has_stamp": pd.BooleanDtype(),
    "parent_candid": pd.Int64Dtype(),
    "diffmaglim": pd.Float32Dtype(),
    "magap": pd.Float32Dtype(),
    "sigmagap": pd.Float32Dtype(),
    "magapbig": pd.Float32Dtype(),
    "sigmagapbig": pd.Float32Dtype(),
    "distnr": pd.Float32Dtype(),
    "nid": pd.Int32Dtype(),
    "rb": pd.Float32Dtype(),
    "rbversion": pd.StringDtype(),
    "drb": pd.Float32Dtype(),
    "drbversion": pd.StringDtype(),
    "magpsf_corr": pd.Float32Dtype(),
    "sigmapsf_corr": pd.Float32Dtype(),
    "sigmapsf_corr_ext": pd.Float32Dtype(),
}


forced_photometry_ztf_db = {
    "oid": pd.Int64Dtype(),
    "sid": pd.Int32Dtype(),
    "tid": pd.Int32Dtype(),
    "pid": pd.Int64Dtype(),
    "band": pd.Int32Dtype(),
    "measurement_id": pd.Int64Dtype(),
    "mjd": pd.Float64Dtype(),
    "ra": pd.Float64Dtype(),
    "dec": pd.Float64Dtype(),
    "mag": pd.Float32Dtype(),
    "e_mag": pd.Float32Dtype(),
    "magpsf_corr": pd.Float32Dtype(),
    "sigmapsf_corr": pd.Float32Dtype(),
    "sigmapsf_corr_ext": pd.Float32Dtype(),
    "isdiffpos": pd.Int32Dtype(),
    "has_stamp": pd.BooleanDtype(),
    "parent_candid": pd.Int64Dtype(),
    "corrected": pd.BooleanDtype(),
    "dubious": pd.BooleanDtype(),
    "field": pd.Int32Dtype(),
    "rcid": pd.Int32Dtype(),
    "rfid": pd.Int64Dtype(),
    "sciinpseeing": pd.Float32Dtype(),
    "scibckgnd": pd.Float32Dtype(),
    "scisigpix": pd.Float32Dtype(),
    "magzpsci": pd.Float32Dtype(),
    "magzpsciunc": pd.Float32Dtype(),
    "magzpscirms": pd.Float32Dtype(),
    "clrcoeff": pd.Float32Dtype(),
    "clrcounc": pd.Float32Dtype(),
    "exptime": pd.Float32Dtype(),
    "adpctdif1": pd.Float32Dtype(),
    "adpctdif2": pd.Float32Dtype(),
    "diffmaglim": pd.Float32Dtype(),
    "programid": pd.Int32Dtype(),
    "procstatus": pd.StringDtype(),
    "distnr": pd.Float32Dtype(),
    "ranr": pd.Float64Dtype(),
    "decnr": pd.Float64Dtype(),
    "magnr": pd.Float32Dtype(),
    "sigmagnr": pd.Float32Dtype(),
    "chinr": pd.Float32Dtype(),
    "sharpnr": pd.Float32Dtype(),
    "new": pd.BooleanDtype(),
}