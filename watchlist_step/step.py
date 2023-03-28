import datetime
import logging
from apf.core.step import GenericStep
from typing import Any, List, Tuple
from db_plugins.db.sql.models import Detection
from db_plugins.db.sql import SQLConnection
from .db.match import (
    format_values_for_query,
    create_insertion_query,
    create_match_query,
)

BASE_RADIUS = 30 / 3600


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
        self.alerts_db_connection = alerts_db_connection or SQLConnection()
        self.alerts_db_connection.connect(config["alert_db_config"]["SQL"])
        self.users_db_connection = users_db_connection or SQLConnection()
        self.users_db_connection.connect(config["users_db_config"]["SQL"])

    def execute(self, messages: list):
        candids = [message["candid"] for message in messages]
        coordinates = self.get_coordinates(candids)
        if len(coordinates) == 0:
            return

        matches = self.match_user_targets(coordinates)
        if len(matches) > 0:
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
        str_values = format_values_for_query(coordinates)
        query = create_match_query(str_values, BASE_RADIUS)
        res = self.users_db_connection.session.execute(query).fetchall()
        return res

    def insert_matches(self, matches: List[Tuple]):
        def tuple_swap(tpl):
            return (tpl[2], tpl[0], tpl[1])

        str_values = format_values_for_query(
            [(*tuple_swap(val), f"{datetime.datetime.now()}") for val in matches]
        )
        query = create_insertion_query(str_values)
        self.users_db_connection.session.execute(query)
        self.users_db_connection.session.commit()
