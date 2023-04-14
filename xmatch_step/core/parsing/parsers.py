import numpy as np
import pandas as pd


# Is this even temporal?
def unparse(data: pd.DataFrame, key: str):
    data = data.copy(deep=True)
    response = []

    if key not in ["detections", "non_detections"]:
        raise NotImplementedError(f"Not implemented unparse for {key} key")

    data = data[key].values
    for dets in data:
        d = pd.DataFrame(dets)
        if d.empty:
            continue
        d = d[d["tid"] == "ZTF"]
        if "extra_fields" in d.columns:
            extra_fields = list(d["extra_fields"].values)
            extra_fields = pd.DataFrame(extra_fields, index=d.index)
            d = d.join(extra_fields)
            d.rename(
                columns={"mag": "magpsf", "e_mag": "sigmapsf"},
                inplace=True,
            )
            d.drop(columns=["extra_fields"], inplace=True)
        response.append(d)

    if len(response) > 0:
        response = pd.concat(response, ignore_index=True)
        response = response.groupby("oid", sort=False).apply(
            lambda x: pd.Series({key: x.to_dict("records")})
        )
        return response

    return pd.DataFrame(columns=[key])


def parse_output(lightcurves: pd.DataFrame, xmatches: pd.DataFrame):
    """Join xmatches with input lightcurves. If xmatch not exists for an object, the value is None. Also generate
    a list of dict as output.
    :param light_curves: Generic messages that contain the light curves (in dataframe)
    :param xmatches: Values of cross-matches (in dataframe)
    :return:
    """
    # Create a new dataframe that contains just two columns `aid` and `xmatches`.
    oid_in = xmatches["oid_in"]  # change to aid for multi stream purposes
    # Temporal code: the oid_in will be removed
    xmatches.drop(
        columns=["ra_in", "dec_in", "col1", "oid_in", "aid_in"],
        inplace=True,
    )
    xmatches.replace({np.nan: None}, inplace=True)
    xmatches = pd.DataFrame(
        {
            "oid_in": oid_in,  # change to aid name for multi stream
            "xmatches": xmatches.apply(
                lambda x: None if x is None else {"allwise": x.to_dict()},
                axis=1,
            ),
        }
    )
    # Join metadata with xmatches
    metadata = lightcurves[["oid", "metadata", "aid", "candid"]].set_index(
        "oid"
    )
    metadata_xmatches = metadata.join(xmatches.set_index("oid_in"))

    # Unparse dets and non dets: means separate detections and non detections by oid. For example if exists a list
    # of oids = [ZTF1, ZTF2] for an aid, te unparse creates a new dataframe of detections and non detections for
    # each oid
    dets = unparse(lightcurves, "detections")
    non_dets = unparse(lightcurves, "non_detections")
    # Join dets and non dets
    dets_nondets = dets.join(non_dets)
    # Join all
    data = metadata_xmatches.join(dets_nondets)

    data.replace({np.nan: None}, inplace=True)
    data.index.names = ["oid"]
    data.reset_index(inplace=True)
    data["candid"] = data["candid"].astype("int64")
    # Transform to a list of dicts
    data = data.to_dict("records")
    return data
