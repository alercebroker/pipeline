from typing import List, Tuple

from watchlist_step.strategies.base import BaseStrategy


class SortingHatStrategy(BaseStrategy):
    def get_coordinates(self, alerts: dict) -> List[tuple]:
        return [(a["ra"], a["dec"], a["oid"], a["candid"]) for a in alerts.values()]

    def get_new_values(
        self, matches: List[tuple], alerts: dict
    ) -> Tuple[List[tuple], List[dict]]:
        new_values = []
        filters = []
        for oid, candid, target_id, filter in matches:
            if len(filter.keys()) == 0 or "sorting_hat" not in filter["fields"].keys():
                continue

            fields = {
                field: alerts[(oid, candid)][field]
                for field in filter["fields"]["sorting_hat"]
            }
            new_values.append((oid, candid, target_id, fields))
            filters.append(filter)

        return new_values, filters
