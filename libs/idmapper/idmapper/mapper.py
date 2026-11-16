"""
Convert ZTF object ids (e.g. "ZTF18abvkqpo") to the int64 oid stored in the multisurvey
database, and back. Other surveys (LSST) already publish integer ids and store them as they come.
"""

import numpy as np

from idmapper.ztf import decode_masterid_for_ztf, encode_ztf_to_masterid_without_survey

# Survey ids as stored in the database (table sid_lut, column object.sid).
# Permanent: every row of object and its related tables carries one of these values.
SURVEY_IDS = {
    "ZTF": 0,
    "LSST DIA Object": 1,
    "LSST SS Object": 2,
}

# A ZTF oid is the encoded ZTF name with a prefix in its top 8 bits. The prefix is sid + 1,
# so every stored ZTF oid has prefix 1.
PREFIX_SHIFT = 63 - 8
ZTF_PREFIX = SURVEY_IDS["ZTF"] + 1


def catalog_oid_to_masterid(
    catalog: str, catalog_oid: str, validate: bool = True
) -> np.int64:
    """Convert a ZTF object id to its oid. Only "ZTF" is supported."""
    if catalog != "ZTF":
        raise ValueError(f"Only ZTF ids need converting, got catalog {catalog!r}.")
    payload = encode_ztf_to_masterid_without_survey(catalog_oid, validate)
    return np.int64((ZTF_PREFIX << PREFIX_SHIFT) + int(payload))


def decode_masterid(masterid: int | np.integer) -> tuple[str, str]:
    """Convert a ZTF oid back to ("ZTF", object id)."""
    masterid = int(masterid)
    prefix = masterid >> PREFIX_SHIFT
    if prefix != ZTF_PREFIX:
        raise ValueError(
            f"{masterid} is not a ZTF oid: prefix {prefix}, expected {ZTF_PREFIX}."
        )
    payload = masterid & ((1 << PREFIX_SHIFT) - 1)
    return "ZTF", decode_masterid_for_ztf(np.int64(payload))
