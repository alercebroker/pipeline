import time
import string
import logging

import pandas as pd

from ..database import MongoConnection
from .database import (
    oid_query,
    conesearch_query,
    update_query,
    insert_empty_objects_to_sql,
)
from copy import deepcopy


CHARACTERS = string.ascii_lowercase

RADIUS = 1.5  # arcsec
BASE = len(CHARACTERS)

logger = logging.getLogger("alerce.SortingHatWizard")


def encode(long_number: int) -> str:
    """
    Encode a long number to string in base 24

    :param long_number: id generated from ra dec
    :return: base 24 of input
    """
    representation = []
    while long_number:
        char = CHARACTERS[long_number % BASE]
        long_number = long_number // BASE
        representation.append(char)
    representation.reverse()
    name = "".join(representation)
    return name


def decode(name: str) -> int:
    """
    Decode a string in base 24 to long integer

    :param name: encoded name in base 24
    :return: decoded name in long integer
    """
    i = 0
    for char in name:
        i = i * BASE + CHARACTERS.index(char)
    return i


def id_generator(ra: float, dec: float) -> int:
    """
    Method that create an identifier of 19 digits given its ra, dec.
    :param ra: right ascension in degrees
    :param dec: declination in degrees
    :return: alerce id
    """
    # 19-Digit ID - two spare at the end for up to 100 duplicates
    aid = 1000000000000000000

    # 2013-11-15 KWS Altered code to fix the negative RA problem
    if ra < 0.0:
        ra += 360.0

    if ra > 360.0:
        ra -= 360.0

    # Calculation assumes Decimal Degrees:
    ra_hh = int(ra / 15)
    ra_mm = int((ra / 15 - ra_hh) * 60)
    ra_ss = int(((ra / 15 - ra_hh) * 60 - ra_mm) * 60)
    ra_ff = int((((ra / 15 - ra_hh) * 60 - ra_mm) * 60 - ra_ss) * 100)

    if dec >= 0:
        h = 1
    else:
        h = 0
        dec = dec * -1

    dec_deg = int(dec)
    dec_mm = int((dec - dec_deg) * 60)
    dec_ss = int(((dec - dec_deg) * 60 - dec_mm) * 60)
    dec_f = int(((((dec - dec_deg) * 60 - dec_mm) * 60) - dec_ss) * 10)

    aid += ra_hh * 10000000000000000
    aid += ra_mm * 100000000000000
    aid += ra_ss * 1000000000000
    aid += ra_ff * 10000000000

    aid += h * 1000000000
    aid += dec_deg * 10000000
    aid += dec_mm * 100000
    aid += dec_ss * 1000
    aid += dec_f * 100
    # transform to str
    return aid


def find_existing_id(db: MongoConnection, alerts: pd.DataFrame):
    """
    The aid column will be assigned the existing alerce id obtained from the database (if found).

    Input:
            oid    aid
        0   A
        1   B
        2   C

    Output:
            oid    aid
        0   A      aid1
        1   B      aid2
        2   C      None
    """
    alertscopy = deepcopy(alerts)
    if "aid" not in alertscopy:
        alertscopy["aid"] = None
    alerts_wo_aid = alertscopy[alertscopy["aid"].isna()]
    if not len(alerts_wo_aid.index):
        logger.debug("0 alerts assigned AID via OID matching")
        return alertscopy

    count = 0
    for oid, group in alerts_wo_aid.groupby("oid"):
        aid = oid_query(db, group["oid"].unique().tolist())
        count += group.index.size if aid else 0
        alerts_wo_aid.loc[group.index, "aid"] = aid
    logger.debug(f"{count} alerts assigned AID via OID matching")

    alertscopy.loc[alerts_wo_aid.index, "aid"] = alerts_wo_aid["aid"]
    return alertscopy


def find_id_by_conesearch(db: MongoConnection, alerts: pd.DataFrame):
    """Assigns aid based on a conesearch in the database.

    Input:
            oid    aid   ra   dec
        0   A      aid1   123   456
        1   B      aid2   123   456
        2   C      None   123   456

    Output:
            oid    aid   ra   dec
        0   A      aid1   123   456
        1   B      aid2   123   456
        2   C      aid3   123   456
    """
    alerts = alerts.copy()
    if "aid" not in alerts:
        alerts["aid"] = None
    alerts_wo_aid = alerts[alerts["aid"].isna()]
    if not len(alerts_wo_aid.index):
        logger.debug("0 alerts assigned AID via cone-search")
        return alerts

    count = 0
    for oid, group in alerts_wo_aid.groupby("oid"):
        aid = conesearch_query(
            db, group["ra"].iloc[0], group["dec"].iloc[0], RADIUS
        )
        count += group.index.size if aid else 0
        alerts_wo_aid.loc[group.index, "aid"] = aid
    logger.debug(f"{count} alerts assigned AID via cone-search")

    alerts.loc[alerts_wo_aid.index, "aid"] = alerts_wo_aid["aid"]
    return alerts


def generate_new_id(alerts: pd.DataFrame):
    """
    Assigns aid column to a dataframe that has oid column.
    The aid column will be a new generated alerce id.

    Input:
            oid    aid   ra   dec
        0   A      aid1   123   456
        1   B      aid2   123   456
        2   C      None   123   456

    Output:
            oid    aid   ra   dec
        0   A      aid1   123   456
        1   B      aid2   123   456
        2   C      ALid   123   456
    """
    alerts = alerts.copy()
    if "aid" not in alerts:
        alerts["aid"] = None
    alerts_wo_aid = alerts[alerts["aid"].isna()]
    if not len(alerts_wo_aid.index):
        logger.debug("No new AIDs created")
        return alerts

    count = 0
    for oid, group in alerts_wo_aid.groupby("oid"):
        id_ = id_generator(group["ra"].iloc[0], group["dec"].iloc[0])
        alerts_wo_aid.loc[group.index, "aid"] = (
            f"AL{time.strftime('%y')}{encode(id_)}"
        )
        count += 1

    logger.debug(
        f"Created {count} new AIDs for {len(alerts_wo_aid.index)} alerts"
    )
    alerts.loc[alerts_wo_aid.index, "aid"] = alerts_wo_aid["aid"]
    return alerts


def insert_empty_objects(
    mongodb: MongoConnection, alerts: pd.DataFrame, psql=None
):
    """
    Inserts an empty entry to the database for every unique _id in the
    alerts dataframe
    :param db: Connection to the database.
    :alerts: Dataframe with alerts.
    """
    objects = alerts[["oid", "aid", "sid", "extra_fields", "ra", "dec", "mjd"]]
    objects = objects.rename(columns={"oid": "_id"}).to_dict("records")
    logger.debug(
        f"Upserting {len(objects)} entries into the Objects collection"
    )
    if mongodb:
        update_query(mongodb, objects)
    if psql:
        insert_empty_objects_to_sql(psql, objects)
