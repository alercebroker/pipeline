from watchlist_step.strategies.base import BaseStrategy

id_to_char = {"g": 1, "r": 2}


class SortingHatStrategy(BaseStrategy):
    def get_coordinates(self, alerts: dict) -> list[tuple]:
        return [(a["ra"], a["dec"], a["oid"], a["candid"]) for a in alerts.values()]

    def get_new_values(self, matches: list[tuple], alerts: dict) -> list[tuple]:
        new_values = []
        for oid, candid, target_id, filter in matches:
            if (
                not filter
                or "fields" not in filter
                or "filters" not in filter
                or "sorting_hat" not in filter["fields"]
            ):
                new_values.append((oid, candid, target_id, {}, filter))
                continue

            fields = {
                field: alerts[(oid, candid)][field]
                for field in filter["fields"]["sorting_hat"]
            }

            if "fid" in fields:
                fields["fid"] = id_to_char[fields["fid"]]

            new_values.append((oid, candid, target_id, fields, filter))

        return new_values
