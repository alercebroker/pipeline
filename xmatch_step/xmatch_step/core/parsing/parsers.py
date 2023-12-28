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
            "oid_in": oid_in,  # change to aid name for multi stream
            "xmatches": xmatches.apply(
                lambda x: None if x is None else {"allwise": x.to_dict()},
                axis=1,
            ),
        }
    ).rename(columns={"oid_in": "oid"})
    xmatches = xmatches.set_index("oid")
    for oid, xmatch in xmatches.iterrows():
        lightcurve_by_oid[oid]["xmatches"] = xmatch["xmatches"]
        lightcurve_by_oid[oid]["candids"] = candids[oid]
    return lightcurve_by_oid
