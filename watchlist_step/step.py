from apf.core.step import GenericStep
import logging
from typing import Any, List, Tuple
from db_plugins.db.sql.models import Detection
from db_plugins.db.sql import SQLConnection
import datetime


class WatchlistStep(GenericStep):
    """WatchlistStep Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(
        self,
        consumer=None,
        alerts_db_connection: SQLConnection = None,
        users_db_connection: SQLConnection = None,
        config=None,
        level=logging.INFO,
        **step_args,
    ):
        super().__init__(consumer, config=config, level=level)
        self.alerts_db_connection = alerts_db_connection
        self.alerts_db_connection.connect(config["alert_db_config"]["SQL"])
        self.users_db_connection = users_db_connection
        self.users_db_connection.connect(config["users_db_config"]["SQL"])

    def execute(self, messages: list):
        candids = [message["candid"] for message in messages]
        coordinates = self.get_coordinates(candids)
        matches = self.match_user_targets(coordinates)
        if len(matches):
            self.insert_matches(matches)

    def get_coordinates(self, candids: List[int]) -> List[Tuple]:
        radecs = (
            self.alerts_db_connection.query(
                Detection.ra, Detection.dec, Detection.oid, Detection.candid
            )
            .filter(Detection.candid.in_(candids))
            .all()
        )
        return radecs

    def match_user_targets(self, coordinates: List[Tuple]) -> List[Tuple]:
        str_values = ",\n".join(
            [f"({val[0]}, {val[1]}, '{val[2]}', '{val[3]}')" for val in coordinates]
        )
        query = (
            """
        WITH positions (ra, dec, oid, candid) AS (
            VALUES %s
        )
        SELECT positions.oid, positions.candid, watchlist_target.id FROM watchlist_target, positions
        WHERE ST_DWITHIN(
            ST_SetSRID(ST_MakePoint(positions.ra, positions.dec), 4035) ,
            ST_SetSRID(ST_MakePoint(watchlist_target.ra, watchlist_target.dec), 4035),
            degrees_to_meters(watchlist_target.sr), true);
        """
            % str_values
        )
        res = self.users_db_connection.session.execute(query).fetchall()
        return res

    def insert_matches(self, matches: List[Tuple]):
        str_values = ",\n".join(
            [
                f"({val[2]}, '{val[0]}', '{val[1]}', '{datetime.datetime.now()}')"
                for val in matches
            ]
        )
        query = (
            """
        INSERT INTO watchlist_match (target, object_id, candid, date) VALUES %s;
        """
            % str_values
        )
        self.users_db_connection.session.execute(query)
