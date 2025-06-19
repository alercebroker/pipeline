import numpy as np
import psycopg2
from psycopg2 import sql


def encode_lsst_to_masterid_without_survey_without_db(
    lsst_oid: np.int64 | int | str,
) -> np.int64:
    """
    Encode LSST object ID to master ID without the survey prefix.

    Parameters
    ----------
    lsst_oid : np.int64 | int | str
        The LSST object ID (diaObjectId).

    Returns
    -------
    np.int64
        The master ID without the survey prefix.
    """
    if isinstance(lsst_oid, np.int64):
        return lsst_oid
    elif isinstance(lsst_oid, int):
        return np.int64(lsst_oid)
    elif isinstance(lsst_oid, str) and lsst_oid.isdigit():
        return np.int64(lsst_oid)
    else:
        raise ValueError(f"Invalid LSST object ID: {lsst_oid}")


def decode_masterid_for_lsst(
    masterid_without_survey: np.int64, db_cursor: psycopg2.extensions.cursor
) -> np.int64:
    """
    Decode a master ID into its components, looking up the original LSST object ID
    in the database.

    Parameters
    ----------
    masterid_without_survey : np.int64
        The master ID without the survey ID.

    Returns
    -------
    np.int64
        The original LSST object ID.
    """
    db_cursor.execute(
        "SELECT lsst_diaObjectId FROM lsst_idmapper WHERE lsst_id_serial = %s",
        (int(masterid_without_survey),),
    )
    result = db_cursor.fetchone()
    if result is not None:
        return np.int64(result[0])
    else:
        raise ValueError(
            f"Master ID {masterid_without_survey} not found in the database"
        )


def encode_lsst_to_masterid_without_survey_with_db(
    lsst_oid: np.int64 | int | str, db_cursor: psycopg2.extensions.cursor
):
    """
    Encode LSST object ID to master ID without the survey prefix,
    using a database stored mapping
    :param lsst_oid: np.int64 | int | str
        The LSST object ID (diaObjectId).
    :return: np.int64
        The master ID without the survey prefix.
    """
    if isinstance(lsst_oid, int):
        pass
    elif isinstance(lsst_oid, np.int64):
        lsst_oid = int(lsst_oid)
    elif isinstance(lsst_oid, str) and lsst_oid.isdigit():
        lsst_oid = int(lsst_oid)
    else:
        raise ValueError(f"Invalid LSST object ID: {lsst_oid}")

    # Check if the LSST object ID is already in the database
    db_cursor.execute(
        sql.SQL(
            'SELECT lsst_id_serial FROM lsst_idmapper WHERE "lsst_diaObjectId" = %s'
        ),
        (lsst_oid,),
    )
    result = db_cursor.fetchone()
    if result is not None:
        lsst_id_serial = np.int64(result[0])
    else:
        # If it doesn't exist, insert it into the database
        db_cursor.execute(
            sql.SQL(
                'INSERT INTO lsst_idmapper ("lsst_diaObjectId") VALUES (%s) RETURNING lsst_id_serial'
            ),
            (lsst_oid,),
        )
        lsst_id_serial = np.int64(db_cursor.fetchone()[0])

    return lsst_id_serial
