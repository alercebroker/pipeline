import numpy as np

from idmapper.lsst import (
    encode_lsst_to_masterid_without_survey_with_db,
    decode_masterid_for_lsst,
)
from idmapper.ztf import encode_ztf_to_masterid_without_survey, decode_masterid_for_ztf

# Constants
SURVEY_IDS = {
    "ZTF": 1,
    "ATLAS": 2,
    "LSST": 3,
    "LS4": 4,
}
SURVEY_PREFIX_LEN_BITS = 8
SURVEY_IDS["MAXSURVEY"] = 2**SURVEY_PREFIX_LEN_BITS - 1

REVERSE_SURVEY_IDS = dict((zip(SURVEY_IDS.values(), SURVEY_IDS.keys())))


def catalog_oid_to_masterid(
    catalog: str,
    catalog_oid: str | np.int64 | int,
    validate: bool = False,
    db_cursor=None,
) -> np.int64:
    """
    Convert a catalog object ID to a master ID.

    Parameters
    ----------
    catalog : str
        The name of the catalog (e.g., "ZTF").
    catalog_oid : str
        The ZTF object ID.
    validate: bool
        If True, validate the ztf_oid before conversion.
    db_cursor: psycopg2.extensions.cursor
        Database cursor for LSST catalog. This parameter is required for LSST.

    Returns
    -------
    str
        The master ID.
    """
    if catalog not in SURVEY_IDS.keys():
        raise ValueError(f"Unsupported catalog: {catalog}")

    # Add the survey ID to the master ID
    master_id = SURVEY_IDS[catalog] << (63 - SURVEY_PREFIX_LEN_BITS)
    master_id = np.int64(master_id)

    if catalog == "ZTF":
        master_id += encode_ztf_to_masterid_without_survey(catalog_oid, validate)
    elif catalog == "LSST":
        if db_cursor is None:
            raise ValueError("db_cursor must be provided for LSST catalog")
        master_id += encode_lsst_to_masterid_without_survey_with_db(
            catalog_oid, db_cursor
        )

    return master_id


def decode_masterid(masterid: np.int64, db_cursor=None) -> tuple[str, str | np.int64]:
    """
    Decode a master ID into its components.

    Parameters
    ----------
    masterid : np.int64
        The master ID.
    db_cursor: psycopg2.extensions.cursor
        Database cursor for LSST catalog. This parameter is required for LSST.

    Returns
    -------
    tuple[str, str]
        The survey of the object and the original oid.
    """
    # Extract the survey from the master ID
    survey_id = masterid >> (63 - SURVEY_PREFIX_LEN_BITS)

    if survey_id in REVERSE_SURVEY_IDS.keys():
        survey = REVERSE_SURVEY_IDS[survey_id]
    else:
        raise ValueError(f"Invalid survey ID: {survey_id}")

    masterid_without_survey = np.bitwise_and(
        masterid, ((1 << (63 - SURVEY_PREFIX_LEN_BITS)) - 1)
    )

    if survey == "ZTF":
        return "ZTF", decode_masterid_for_ztf(masterid_without_survey)

    elif survey == "LSST":
        if db_cursor is None:
            raise ValueError("db_cursor must be provided for LSST catalog")
        return "LSST", decode_masterid_for_lsst(masterid_without_survey, db_cursor)

    else:
        raise ValueError(f"Unsupported survey ID: {survey_id}")
