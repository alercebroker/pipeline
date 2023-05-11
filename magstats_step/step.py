import json
import logging
from typing import List

from apf.core.step import GenericStep, get_class

from magstats_step.core import MagnitudeStatistics, ObjectStatistics


class MagstatsStep(GenericStep):
    def __init__(
        self,
        config,
        level=logging.INFO,
        **step_args,
    ):
        super().__init__(config=config, level=level, **step_args)
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

    def execute(self, messages: dict):
        obj_calculator = ObjectStatistics(messages["detections"])
        stats = obj_calculator.generate_statistics(self.excluded)

        stats = stats.to_dict("index")

        magstats_calculator = MagnitudeStatistics(**messages)
        magstats = magstats_calculator.generate_statistics(self.excluded).reset_index()
        magstats = magstats.set_index("aid")
        for aid in stats:
            try:
                stats[aid]["magstats"] = magstats.loc[aid].to_dict("records")
            except TypeError:
                stats[aid]["magstats"] = [magstats.loc[aid].to_dict()]

        return stats

    def produce_scribe(self, result: dict):
        for aid, stats in result.items():
            command = {
                "collection": "object",
                "type": "update",
                "criteria": {"_id": aid},
                "data": stats | {"loc": {"type": "Point", "coordinates": [stats["meanra"] - 180, stats["meandec"]]}},
                "options": {"upsert": True},
            }
            self.scribe_producer.produce({"payload": json.dumps(command)})

    def post_execute(self, result: dict):
        self.produce_scribe(result)
        return result
