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
    xmatches = pd.DataFrame(
        {
            "oid_in": oid_in,
            "xmatches": xmatches.apply(
                lambda x: None if x is None else {"allwise": x.to_dict()},
                axis=1,
            ),
        }
    ).rename(columns={"oid_in": "oid"})
    xmatches = xmatches.set_index("oid")
    xmatches.index.drop_duplicates(keep="last")
    result = []
    for oid in lightcurve_by_oid:
        lightcurve_by_oid[oid]["xmatches"] = xmatches.loc[oid][
            "xmatches"
        ].to_dict()[oid]
        lightcurve_by_oid[oid]["candid"] = candids[oid]
        lightcurve_by_oid[oid]["oid"] = oid
        result.append(lightcurve_by_oid[oid])
    return result
