import time
import string

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from db_plugins.db.mongo import DatabaseConnection

from .database import oid_query, conesearch_query


CHARACTERS = string.ascii_lowercase

RADIUS = 1.5  # arcsec
BASE = len(CHARACTERS)


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


def internal_cross_match(
    data: pd.DataFrame, ra_col="ra", dec_col="dec"
) -> pd.DataFrame:
    """
    Do an internal cross-match in data input (batch vs batch) to get the closest objects. This method uses
    cKDTree class to get the nearest object. Returns a new dataframe with another column named tmp_id to
    reference unique objects :param data: alerts in a dataframe :param ra_col: how the ra column is called in
    data :param dec_col: how the dec column is called in data :return:
    """
    data = data.copy()
    radius = RADIUS / 3600
    values = data[[ra_col, dec_col]].to_numpy()
    tree = cKDTree(values)
    sdm = tree.sparse_distance_matrix(
        tree, radius, output_type="coo_matrix"
    )  # get sparse distance matrix
    # Get the matrix representation -> rows x cols
    matrix = sdm.toarray()

    # Put the index as a tmp_id
    data["tmp_id"] = data.index
    # Get unique object_ids
    oids = data["oid"].unique()
    for index, oid in enumerate(oids):  # join the same objects
        indexes = data[data["oid"] == oid].index  # get all indexes of this oid
        if (
            len(indexes) > 1
        ):  # if exists an oid with more than 1 occurrences put the same tmp_id
            data.loc[indexes, "tmp_id"] = index
            # for remove neighbors get the combination or all indexes of the same object
            a, b = np.meshgrid(indexes, indexes, sparse=True)
            # remove in adjacency matrix
            matrix[a, b] = 0
    while matrix.sum():  # while exists matches
        matches = np.count_nonzero(matrix, axis=1)  # count of matches per node (row)
        # get rows with max matches (can be more than 1)
        max_matches = np.argwhere(matches == matches.max(axis=0)).flatten()
        dist_matches = matrix[max_matches].sum(
            axis=1
        )  # compute sum of distance of each element in max_matches
        min_dist = np.argmin(dist_matches)  # get index of min sum of distance
        node = max_matches[
            min_dist
        ]  # chosen node: with most matches and the least distance
        neighbours = matrix[node, :]  # get all neighbours of the node
        neighbours_indexes = np.flatnonzero(neighbours)  # get indexes of the neighbours
        data.loc[neighbours_indexes, "tmp_id"] = data["tmp_id"][
            node
        ]  # put tmp_id of the neighbours
        matrix[neighbours_indexes, :] = 0  # turn off neighbours
        matrix[:, neighbours_indexes] = 0
    return data


def find_existing_id(db: DatabaseConnection, alerts: pd.DataFrame):
    """
    Assigns aid column to a dataframe that has tmp_id column.
    The aid column will be the existing alerce id obtained by the injected database_id_getter.
    If no id is found, the resulting aid column will contain null values.

    Input:
            oid    tmp_id
        0   A      X
        1   B      X
        2   C      Y

    Output:
            oid    tmp_id
        0   A      X      aid1
        1   B      X      aid1
        2   C      Y      None
    """
    def _find_existing_id(data: pd.DataFrame):
        oids = data["oid"].unique().tolist()
        aid = oid_query(db, oids)
        return pd.Series({"aid": aid})

    tmp_id_aid = alerts.groupby("tmp_id").apply(_find_existing_id)
    return alerts.join(tmp_id_aid, on="tmp_id")


def find_id_by_conesearch(db: DatabaseConnection, alerts: pd.DataFrame):
    """
    Assigns aid column to a dataframe that has tmp_id column.
    The aid column will be the existing alerce id obtained by the injected conesearch_id_getter.
    If no id is found, the resulting aid column will contain null values.

    Input:
            oid    tmp_id  aid   ra   dec
        0   A      X     aid1   123   456
        1   B      X     aid1   123   456
        2   C      Y     None   123   456

    Output:
            oid    tmp_id  aid   ra   dec
        0   A      X     aid1   123   456
        1   B      X     aid1   123   456
        2   C      Y     aid2   123   456
    """
    alerts_copy = alerts.copy()
    alerts_without_aid = alerts_copy[alerts_copy["aid"].isna()]
    if len(alerts_without_aid) == 0:
        return alerts_copy

    def _find_id_by_conesearch(data: pd.DataFrame):
        first_alert = data.iloc[0]
        ra = first_alert["ra"]
        dec = first_alert["dec"]

        near_objects = conesearch_query(db, ra, dec, RADIUS)
        aid = None
        if len(near_objects):
            aid = near_objects[0]["aid"]
        return pd.Series({"aid": aid})

    tmp_id_aid = alerts_without_aid.groupby("tmp_id").apply(_find_id_by_conesearch)
    alerts_copy = alerts_copy.join(tmp_id_aid, on="tmp_id", rsuffix="_found")
    alerts_copy["aid"] = alerts_copy["aid"].fillna(alerts_copy["aid_found"])
    alerts_copy = alerts_copy.drop(columns=["aid_found"])
    return alerts_copy


def generate_new_id(alerts: pd.DataFrame):
    """
    Assigns aid column to a dataframe that has tmp_id column.
    The aid column will be a new generated alerce id.

    Input:
            oid    tmp_id  aid   ra   dec
        0   A      X     aid1   123   456
        1   B      X     aid1   123   456
        2   C      Y     None   123   456

    Output:
            oid    tmp_id  aid   ra   dec
        0   A      X     aid1   123   456
        1   B      X     aid1   123   456
        2   C      Y     ALid   123   456
    """
    alerts_copy = alerts.copy()
    alerts_without_aid = alerts_copy[alerts_copy["aid"].isna()]
    if len(alerts_without_aid) == 0:
        return alerts_copy

    def _generate_new_id(data: pd.DataFrame):
        first_alert = data.iloc[0]
        aid = id_generator(first_alert["ra"], first_alert["dec"])  # this is the long id
        aid = encode(aid)  # this is the long id encoded to string id
        year = time.strftime("%y")  # get year in short form. e.g: 21 means 2021
        aid = f"AL{year}{aid}"  # put prefix of ALeRCE to string id. e.g: 'AL{year}{long_id}'
        return pd.Series({"aid": aid})

    tmp_id_aid = alerts_without_aid.groupby("tmp_id").apply(_generate_new_id)
    alerts_copy = alerts_copy.join(tmp_id_aid, on="tmp_id", rsuffix="_found")
    alerts_copy["aid"] = alerts_copy["aid"].fillna(alerts_copy["aid_found"])
    alerts_copy = alerts_copy.drop(columns=["aid_found"])
    return alerts_copy
