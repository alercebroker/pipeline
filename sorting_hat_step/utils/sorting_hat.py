import numpy as np
import pandas as pd
import time
import string

from scipy.spatial import cKDTree
from typing import List

from db_plugins.db.mongo.models import Object
from db_plugins.db.mongo.connection import MongoConnection

CHARACTERS = string.ascii_lowercase


# https://media.giphy.com/media/JDAVoX2QSjtWU/giphy.gif
class SortingHat:
    def __init__(self, db: MongoConnection, radius: float = 1.5):
        self.radius = radius
        self.db = db
        # Values from WGS 84
        self.a = 6378137.000000000000  # Semi-major axis of Earth
        self.e = 0.081819190842600  # eccentricity
        self.angle = np.radians(1.0)
        self.base = len(CHARACTERS)

    def wgs_scale(self, lat: float) -> float:
        """
        Get scaling to convert degrees to meters at a given geodetic latitude (declination)
        :param lat: geodetic latitude (declination)
        :return:
        """
        # Compute radius of curvature along meridian (see https://en.wikipedia.org/wiki/Meridian_arc)
        rm = self.a * (1 - np.power(self.e, 2)) / np.power(
            (1 - np.power(self.e, 2) * np.power(np.sin(np.radians(lat)), 2)), 1.5)
        # Compute length of arc at this latitude (meters/degree)
        arc = rm * self.angle
        return arc

    def cone_search(self, ra: float, dec: float) -> List[dict]:
        """
        Cone search to database given a ra, dec and radius. Returns a list of objects sorted by distance.
        :param ra: right ascension
        :param dec: declination
        :return:
        """
        radius = self.radius / 3600
        scaling = self.wgs_scale(dec)
        meter_radius = radius * scaling
        lon, lat = ra - 180., dec
        objects = self.db.query(model=Object)
        cursor = objects.find(
            {
                'loc': {
                    '$nearSphere': {
                        '$geometry':
                            {
                                'type': 'Point',
                                'coordinates': [lon, lat]
                            },
                        '$maxDistance': meter_radius,
                    }
                },
            },
            {
                "aid": 1  # only return alerce_id
            }
        )
        spatial = [i for i in cursor]
        return spatial

    def oid_query(self, oid: list) -> int or None:
        """
        Query to database if the oids has an alerce_id
        :param oid: oid of any survey
        :return: existing aid if exists else is None
        """
        objects = self.db.query(model=Object)
        cursor = objects.find(
            {
                "oid": {
                    "$in": oid
                }
            },
            {
                "_id": 0,
                "aid": 1
            }
        )
        data = [i["aid"] for i in cursor]
        if len(data):
            return data[0]
        return None

    def encode(self, long_number: int) -> str:
        """
        Encode a long number to string in base 24

        :param long_number: id generated from ra dec
        :return: base 24 of input
        """
        representation = []
        while long_number:
            char = CHARACTERS[long_number % self.base]
            long_number = long_number // self.base
            representation.append(char)
        representation.reverse()
        name = ''.join(representation)
        return name

    def decode(self, name: str) -> int:
        """
        Decode a string in base 24 to long integer

        :param name: encoded name in base 24
        :return: decoded name in long integer
        """
        i = 0
        for char in name:
            i = i * self.base + CHARACTERS.index(char)
        return i

    @classmethod
    def id_generator(cls, ra: float, dec: float) -> int:
        """
        Method that create an identifier of 19 digits given its ra, dec.
        :param ra: right ascension in degrees
        :param dec: declination in degrees
        :return: alerce id
        """
        # 19 Digit ID - two spare at the end for up to 100 duplicates
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

        aid += (ra_hh * 10000000000000000)
        aid += (ra_mm * 100000000000000)
        aid += (ra_ss * 1000000000000)
        aid += (ra_ff * 10000000000)

        aid += (h * 1000000000)
        aid += (dec_deg * 10000000)
        aid += (dec_mm * 100000)
        aid += (dec_ss * 1000)
        aid += (dec_f * 100)
        # transform to str
        return aid

    def internal_cross_match(self, data: pd.DataFrame, ra_col="ra", dec_col="dec") -> pd.DataFrame:
        """
        Do an internal cross-match in data input (batch vs batch) to get the closest objects. This method uses
        cKDTree class to get the nearest object. Returns a new dataframe with another column named tmp_id to
        reference unique objects :param data: alerts in a dataframe :param ra_col: how the ra column is called in
        data :param dec_col: how the dec column is called in data :return:
        """
        data = data.copy()
        radius = self.radius / 3600
        values = data[[ra_col, dec_col]].to_numpy()
        tree = cKDTree(values)
        sdm = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")  # get sparse distance matrix
        # Get the matrix representation -> rows x cols
        matrix = sdm.toarray()

        # Put the index as a tmp_id
        data["tmp_id"] = data.index
        # Get unique object_ids
        oids = data["oid"].unique()
        for index, oid in enumerate(oids):  # join the same objects
            indexes = data[data["oid"] == oid].index  # get all indexes of this oid
            if len(indexes) > 1:  # if exists an oid with more than 1 occurrences put the same tmp_id
                data.loc[indexes, "tmp_id"] = index
                # for remove neighbors get the combination or all indexes of the same object
                a, b = np.meshgrid(indexes, indexes, sparse=True)
                # remove in adjacency matrix
                matrix[a, b] = 0
        while matrix.sum():  # while exists matches
            matches = np.count_nonzero(matrix, axis=1)  # count of matches per node (row)
            # get rows with max matches (can be more than 1)
            max_matches = np.argwhere(matches == matches.max(axis=0)).flatten()
            dist_matches = matrix[max_matches].sum(axis=1)  # compute sum of distance of each element in max_matches
            min_dist = np.argmin(dist_matches)  # get index of min sum of distance
            node = max_matches[min_dist]  # chosen node: with most matches and the least distance
            neighbours = matrix[node, :]  # get all neighbours of the node
            neighbours_indexes = np.flatnonzero(neighbours)  # get indexes of the neighbours
            data.loc[neighbours_indexes, "tmp_id"] = data["tmp_id"][node]  # put tmp_id of the neighbours
            matrix[neighbours_indexes, :] = 0  # turn off neighbours
            matrix[:, neighbours_indexes] = 0
        return data

    def _to_name(self, group_of_alerts: pd.DataFrame) -> pd.Series:
        """
        Generate alerce_id to a group of alerts of the same object. This method has three options:
        1) First Hit: The alert has an oid existing in database
        2) Second Hit: The alert has a ra, dec closest to object in database (radius of 1.5")
        3) Miss: Create a new aid given its ra, dec
        :param group_of_alerts: alerts of the same object.
        :return:
        """
        oids = group_of_alerts["oid"].unique().tolist()
        first_alert = group_of_alerts.iloc[0]
        # 1) First Hit: Exists at least one aid to this oid
        existing_oid = self.oid_query(oids)
        if existing_oid:
            aid = existing_oid
        else:
            # 2) Second Hit: cone search return objects sorted. So first response is closest.
            near_objects = self.cone_search(first_alert["ra"], first_alert["dec"])
            if len(near_objects):
                aid = near_objects[0]["aid"]
            # 3) Miss generate a new ALeRCE identifier
            else:
                aid = self.id_generator(first_alert["ra"], first_alert["dec"])  # this is the long id
                aid = self.encode(aid)  # this is the long id encoded to string id
                year = time.strftime('%y')  # get year in short form. e.g: 21 means 2021
                aid = f"AL{year}{aid}"  # put prefix of ALeRCE to string id. e.g: 'AL{year}{long_id}'
        response = {"aid": aid}
        return pd.Series(response)

    def to_name(self, alerts: pd.DataFrame) -> pd.DataFrame:
        """
        Generate an alerce_id to a batch of alerts given its oid, ra, dec and radius.
        :param alerts: Dataframe of alerts
        :return: Dataframe of alerts with a new column called `aid` (alerce_id)
        """
        # Internal cross match that identifies same objects in own batch: create a new column named 'tmp_id'
        alerts = self.internal_cross_match(alerts)
        # Interaction with database: group all alerts with the same tmp_id and find/create alerce_id
        tmp_id_aid = alerts.groupby("tmp_id").apply(self._to_name)
        # Join the tuple tmp_id-aid with batch of alerts
        alerts = alerts.set_index("tmp_id").join(tmp_id_aid)
        # Remove column tmp_id (really is an index) forever
        alerts.reset_index(inplace=True, drop=True)
        return alerts
