import json
import numpy as np
from typing import List
from apf.core.step import GenericStep, get_class
from magstats_step.core import MagnitudeStatistics, ObjectStatistics

from pprint import pprint


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
        stats = obj_calculator.generate_statistics(self.excluded).replace({np.nan: None})
        stats = stats.to_dict("index")

        magstats_calculator = MagnitudeStatistics(**messages)
        magstats = magstats_calculator.generate_statistics(self.excluded).reset_index()
        magstats = magstats.set_index("aid").replace({np.nan: None})
        for aid in stats:
            try:
                stats[aid]["magstats"] = magstats.loc[aid].to_dict("records")
            except TypeError:
                stats[aid]["magstats"] = [magstats.loc[aid].to_dict()]

        return stats

    def _execute_ztf(self, messages: dict):
        ztf_detections = list(filter(lambda d: d["sid"] == "ZTF", messages["detections"]))
        obj_calculator = ObjectStatistics(ztf_detections)
        stats = obj_calculator.generate_statistics(self.excluded).replace({np.nan: None})
        stats = stats.to_dict("index")

        magstats_calculator = MagnitudeStatistics(detections=ztf_detections, non_detections=messages["non_detections"])
        magstats = magstats_calculator.generate_statistics(self.excluded).reset_index()
        magstats = magstats.set_index("aid").replace({np.nan: None})
        for aid in stats:
            try:
                stats[aid]["magstats"] = magstats.loc[aid].to_dict("records")
            except TypeError:
                stats[aid]["magstats"] = [magstats.loc[aid].to_dict()]

        return stats

    def execute(self, messages: dict):
        stats = {}
        stats["multistream"] = self._execute(messages)
        stats["ztf"] = self._execute_ztf(messages)
        return stats

    # it seems that we'll have to produce different commands in this
    def produce_scribe(self, result: dict):
        for aid, stats in result.items():
            command = {
                "collection": "object",
                "type": "update",
                "criteria": {"_id": aid},
                "data": stats
                | {
                    "loc": {
                        "type": "Point",
                        "coordinates": [stats["meanra"] - 180, stats["meandec"]],
                    }
                },
                "options": {"upsert": True},
            }
            self.scribe_producer.produce({"payload": json.dumps(command)})

    def produce_scribe_ztf(self, result: dict):
        for stats in result.values():
            stats = {k: v for k, v in stats.items() if k not in ["tid", "sid"]}
            oids = stats.pop("oid")
            commands = [
                {
                    "collection": "magstats",
                    "type": "upsert",
                    "criteria": {"oid": oid},
                    "data": stats,
                }
                for oid in oids
            ]

            for command in commands:
                pprint(command)
                self.scribe_producer.produce({"payload": json.dumps(command)})

    def post_execute(self, result: dict):
        self.produce_scribe(result["multistream"])
        self.produce_scribe_ztf(result["ztf"])
        return {}
