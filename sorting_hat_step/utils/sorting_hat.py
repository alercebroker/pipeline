import numpy as np
import pandas as pd
from typing import List

from db_plugins.db.mongo.models import Object
from db_plugins.db.mongo.connection import MongoConnection


# https://media.giphy.com/media/JDAVoX2QSjtWU/giphy.gif
class SortingHat:
    def __init__(self, db: MongoConnection):
        self.db = db

    @classmethod
    def wgs_scale(cls, lat: float) -> float:
        """
        Get scaling to convert degrees to meters at a given geodetic latitude (declination)
        :param lat: geodetic latitude (declination)
        :return:
        """

        # Values from WGS 84
        a = 6378137.000000000000  # Semi-major axis of Earth
        e = 0.081819190842600  # eccentricity
        angle = np.radians(1.0)

        # Compute radius of curvature along meridian (see https://en.wikipedia.org/wiki/Meridian_arc)
        rm = a * (1 - np.power(e, 2)) / np.power((1 - np.power(e, 2) * np.power(np.sin(np.radians(lat)), 2)), 1.5)

        # Compute length of arc at this latitude (meters/degree)
        arc = rm * angle
        return arc

    def cone_search(self, ra: float, dec: float, radius: float = 1.5) -> List[dict]:
        """
        Cone search to database given a ra, dec and radius. Returns a list of objects sorted by distance.
        :param ra: right ascension
        :param dec: declination
        :param radius:
        :return:
        """
        radius = radius / 3600
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

    def oid_query(self, oid: str) -> int or None:
        """
        Query to database if the oids has an alerce_id
        :param oid: oid of any survey
        :return: existing aid if exists else is None
        """
        objects = self.db.query(model=Object)
        cursor = objects.find(
            {
                "oid": oid
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

        return aid

    def _to_name(self, group_of_alerts: pd.Series) -> pd.Series:
        """
        Generate alerce_id to a group of alerts of the same object. This method has three options:
        1) First Hit: The alert has an oid existing in database
        2) Second Hit: The alert has a ra, dec closest to object in database (radius of 1.5")
        3) Miss: Create a new aid given its ra, dec
        :param group_of_alerts: alerts of the same object.
        :return:
        """
        first_alert = group_of_alerts.iloc[0]
        oid = first_alert["oid"]
        # 1) First Hit: Exists at least one aid to this oid
        existing_oid = self.oid_query(oid)
        if existing_oid:
            aid = existing_oid
        else:
            # 2) Second Hit: cone search return objects sorted. So first response is closest.
            near_objects = self.cone_search(first_alert["ra"], first_alert["dec"])
            if len(near_objects):
                aid = near_objects[0]["aid"]
            # 3) Miss generate a new ALeRCE identifier
            else:
                aid = self.id_generator(first_alert["ra"], first_alert["dec"])
        response = {"aid": aid}
        return pd.Series(response)

    def to_name(self, alerts: pd.DataFrame) -> pd.DataFrame:
        """
        Generate an alerce_id to a batch of alerts given its oid, ra, dec and radius.
        :param alerts: Dataframe of alerts
        :return:
        """
        # Internal cross match here...
        #####################

        # Interaction with database: group all alerts with the same oid (is possible get more than
        # 1 alert with same oid in one batch)
        oid_aid = alerts.groupby("oid").apply(self._to_name)
        # Join the tuple oid-aid with batch of alerts
        alerts = alerts.set_index("oid").join(oid_aid)
        alerts.reset_index(inplace=True)
        alerts["aid"] = alerts["aid"].astype(np.long)
        return alerts
