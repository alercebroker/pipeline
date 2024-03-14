import numpy as np
import pandas as pd


def parse_output(
    xmatches: pd.DataFrame, lightcurve_by_oid: dict, candids: dict
):
    """Join xmatches with input lightcurves.

    If xmatch not exists for an object, the value is None.
    """
    oid_in = xmatches["oid_in"]
    xmatches = xmatches.drop(
        columns=["ra_in", "dec_in", "col1", "oid_in"],
    ).replace({np.nan: None})
    xmatches = (
        pd.DataFrame(
            {
                "oid_in": oid_in,
                "xmatches": xmatches.apply(
                    lambda x: None if x is None else {"allwise": x.to_dict()},
                    axis=1,
                ),
            }
        )
        .rename(columns={"oid_in": "oid", "aid_in": "aid"})
        .drop_duplicates("oid", keep="last")
        .set_index("oid")
    )
    result = []
    for oid in lightcurve_by_oid:
        try:
            lightcurve_by_oid[oid]["xmatches"] = xmatches.loc[oid]["xmatches"]
        except KeyError:
            lightcurve_by_oid[oid]["xmatches"] = None
        lightcurve_by_oid[oid]["candid"] = candids[oid]
        lightcurve_by_oid[oid]["oid"] = oid
        result.append(lightcurve_by_oid[oid])
    return result
