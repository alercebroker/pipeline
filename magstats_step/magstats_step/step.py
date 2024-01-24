import json
import numpy as np
import pandas as pd
from typing import List
from apf.core.step import GenericStep, get_class
from magstats_step.core import MagnitudeStatistics, ObjectStatistics


class MagstatsStep(GenericStep):
    def __init__(
        self,
        config,
        **step_args,
    ):
        super().__init__(config=config, **step_args)
        self.excluded = set(config["EXCLUDED_CALCULATORS"])
        cls = get_class(config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(config["SCRIBE_PRODUCER_CONFIG"])

    @classmethod
    def pre_execute(cls, messages: List[dict]) -> dict:
        detections, non_detections = [], []
        for msg in messages:
            detections.extend(msg["detections"])
            non_detections.extend(msg["non_detections"])
        return {"detections": detections, "non_detections": non_detections}

    def _execute(self, messages: dict):
        obj_calculator = ObjectStatistics(messages["detections"])
        stats = obj_calculator.generate_statistics(self.excluded).replace(
            {np.nan: None}
        )
        stats = stats.to_dict("index")
        magstats_calculator = MagnitudeStatistics(**messages)
        magstats = magstats_calculator.generate_statistics(
            self.excluded
        ).reset_index()
        magstats = magstats.set_index("oid").replace({np.nan: None})
        for oid in stats:
            self.parse_magstats_result(magstats, oid, stats)
        return stats

    def parse_magstats_result(
        self, magstats: pd.DataFrame, oid: str, stats: dict
    ) -> dict:
        """Adds magstats in the correct format for the oid
        **Note**: Updates stats dictionary in place
        """
        magstats_by_oid = magstats.loc[oid]
        if isinstance(magstats_by_oid, pd.Series):
            # when calling to_dict on a series
            # the result is a single dict
            # we then ensure that it is always a list of dict
            stats[oid]["magstats"] = [(magstats_by_oid.to_dict())]
        elif isinstance(magstats_by_oid, pd.DataFrame):
            # when calling to_dict on a dataframe with orient=records
            # the result should be already a list of dicts
            stats[oid]["magstats"] = magstats_by_oid.to_dict(orient="records")
        else:
            raise TypeError(f"Unknown magstats type {type(magstats_by_oid)}")

    def _execute_ztf(self, messages: dict):
        ztf_detections = list(
            filter(lambda d: d["sid"] == "ZTF", messages["detections"])
        )
        if not len(ztf_detections):
            return {}
        obj_calculator = ObjectStatistics(ztf_detections)
        stats = obj_calculator.generate_statistics(self.excluded).replace(
            {np.nan: None}
        )
        stats = stats.to_dict("index")
        magstats_calculator = MagnitudeStatistics(
            detections=ztf_detections,
            non_detections=messages["non_detections"],
        )
        magstats = magstats_calculator.generate_statistics(
            self.excluded
        ).reset_index()
        magstats = magstats.set_index("oid").replace({np.nan: None})
        for oid in stats:
            self.parse_magstats_result(magstats, oid, stats)
        return stats

    def execute(self, messages: dict):
        stats = {}
        stats["multistream"] = self._execute(messages)
        stats["ztf"] = self._execute_ztf(messages)
        return stats

    def produce_scribe(self, result: dict):
        for oid, stats in result.items():
            command = {
                "collection": "object",
                "type": "update",
                "criteria": {"_id": oid},
                "data": stats
                | {
                    "loc": {
                        "type": "Point",
                        "coordinates": [
                            stats["meanra"] - 180,
                            stats["meandec"],
                        ],
                    }
                },
                "options": {"upsert": True},
            }
            self.scribe_producer.produce({"payload": json.dumps(command)})

    def produce_scribe_ztf(self, result: dict):
        commands = []
        for oid, stats in result.items():
            commands.append(
                {
                    "collection": "magstats",
                    "type": "upsert",
                    "criteria": {"_id": oid},
                    "data": stats,
                }
            )
        for command in commands:
            self.scribe_producer.produce({"payload": json.dumps(command)})

    def post_execute(self, result: dict):
        self.produce_scribe(result["multistream"])
        self.produce_scribe_ztf(result["ztf"])
        return {}
