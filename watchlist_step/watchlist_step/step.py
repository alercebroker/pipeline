import datetime
import itertools
import json
import logging
from typing import List

from apf.consumers import KafkaConsumer
from apf.core.step import GenericStep

from .db.connection import PsqlDatabase
from .db.match import (
    create_insertion_query,
    create_match_query,
    update_for_notification,
    update_match_query,
)
from .filters import satisfies_filter
from .strategies import get_strategy

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
        config: dict,
        strategy_name: str = "",
        level=logging.INFO,
    ):
        super().__init__(config=config, level=level)
        self.users_db = PsqlDatabase(config["PSQL_CONFIG"])
        self.strategy = get_strategy(strategy_name)

    def insert_matches(self, matches: List[tuple]):
        values = [
            (m[2], m[0], m[1], json.dumps({}), datetime.datetime.now()) for m in matches
        ]

        query = create_insertion_query()

        with self.users_db.conn() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(query, values)

    def match_user_targets(self, coordinates: List[tuple]) -> List[tuple]:
        query = create_match_query(len(coordinates), BASE_RADIUS)

        res = []
        with self.users_db.conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, [item for coord in coordinates for item in coord])
                res = cursor.fetchall()
        return res

    def update_match_values(self, new_values: List[tuple]) -> List[tuple]:
        values = [
            {
                "oid": oid,
                "candid": candid,
                "target_id": target_id,
                "values": json.dumps(values),
                "filter": filter,
            }
            for oid, candid, target_id, values, filter in new_values
        ]

        query = update_match_query()

        updated = []
        with self.users_db.conn() as conn:
            with conn.cursor() as cursor:
                for value in values:
                    cursor.execute(query, value)
                    res = cursor.fetchone()
                    if res is not None:
                        updated.append((*res, value["filter"]))
        return updated

    def get_to_notify(self, updated_values: list[tuple]) -> list[tuple]:
        to_notify = []
        for oid, candid, target_id, values, filter in updated_values:
            if not filter or "fields" not in filter or "filters" not in filter:
                to_notify.append((oid, candid, target_id))
                continue
            available_fields = set(values.keys())
            requiered_fields = set(itertools.chain(*filter["fields"].values()))

            if available_fields.issuperset(requiered_fields):
                satisfies_all_filters = satisfies_filter(
                    values, "and", {"filters": filter["filters"]}
                )
                if satisfies_all_filters:
                    to_notify.append((oid, candid, target_id))

        return to_notify

    def mark_for_notification(self, to_notify: list[tuple]):
        values = [
            {"oid": oid, "candid": candid, "target_id": target_id}
            for oid, candid, target_id in to_notify
        ]

        query = update_for_notification()

        with self.users_db.conn() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(query, values)

    def execute(self, message: List[dict]):
        alerts = {(m["oid"], m["candid"]): m for m in message}
        coordinates = self.strategy.get_coordinates(alerts)

        matches = self.match_user_targets(coordinates)

        if len(matches) > 0:
            self.insert_matches(matches)

        new_values = self.strategy.get_new_values(matches, alerts)
        updated_values = self.update_match_values(new_values)

        to_notify = self.get_to_notify(updated_values)
        if len(to_notify) > 0:
            self.mark_for_notification(to_notify)

        return []

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
