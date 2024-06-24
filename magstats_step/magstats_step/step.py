import json
import numpy as np
import pandas as pd
from typing import List
from apf.core.step import GenericStep, get_class
from magstats_step.core import MagnitudeStatistics, ObjectStatistics
from importlib.metadata import version


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
        self.version = version("magstats-step")

    @classmethod
    def pre_execute(cls, messages: List[dict]) -> dict:
        detections, non_detections = [], []
        for msg in messages:
            # code to add mjdendref to detections fields from extra fields
            for det in msg["detections"]:
                jdendref = det["extra_fields"].get("jdendref", None)
                if jdendref:
                    det["mjdendref"] = jdendref - 2400000.5
            detections.extend(msg["detections"])
            non_detections.extend(msg["non_detections"])
        return {"detections": detections, "non_detections": non_detections}

    def _calculate_object_statistics(self, detections) -> dict:
        obj_calculator = ObjectStatistics(detections)
        stats = obj_calculator.generate_statistics(self.excluded).replace(
            {np.nan: None}
        )
        assert stats.index.name == "oid"
        return stats.to_dict("index")

    def _calculate_magstats(self, detections, non_detections) -> pd.DataFrame:
        magstats_calculator = MagnitudeStatistics(
            detections=detections, non_detections=non_detections
        )
        stats = (
            magstats_calculator.generate_statistics(self.excluded)
            .reset_index()
            .set_index("oid")
            .replace({np.nan: None})
        )
        return stats

    def execute(self, message: dict):
        stats = self._calculate_object_statistics(message["detections"])
        magstats = self._calculate_magstats(
            message["detections"], message["non_detections"]
        )
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
        # magstats_by_oid could have multiple fid values,
        # so the loc could potentially return a dataframe with a row
        # for each fid
        if isinstance(magstats_by_oid, pd.Series):
            # when calling to_dict on a series
            # the result is a single dict
            # we then ensure that it is always a list of dict
            stats[oid]["magstats"] = [magstats_by_oid.to_dict()]
        elif isinstance(magstats_by_oid, pd.DataFrame):
            # when calling to_dict on a dataframe with orient=records
            # the result should be already a list of dicts
            stats[oid]["magstats"] = magstats_by_oid.to_dict(orient="records")
        else:
            raise TypeError(f"Unknown magstats type {type(magstats_by_oid)}")

    def produce_scribe(self, result: dict):
        num_commands = len(result)
        count = 0
        for oid, stats in result.items():
            count += 1
            stats["step_version"] = self.version
            stats["loc"] = {
                "type": "Point",
                "coordinates": [stats["meanra"] - 180, stats["meandec"]],
            }
            command = {
                "collection": "object",
                "type": "update",
                "criteria": {"_id": oid},
                "data": stats,
                "options": {"upsert": True},
            }
            flush = count == num_commands
            self.scribe_producer.produce(
                {"payload": json.dumps(command)}, flush=flush, key=oid
            )

    def produce_scribe_ztf(self, result: dict):
        num_commands = len(result)
        count = 0
        for oid, stats in result.items():
            count += 1
            stats["step_id_corr"] = self.version
            command = {
                "collection": "magstats",
                "type": "upsert",
                "criteria": {"_id": oid},
                "data": stats,
            }
            flush = count == num_commands
            self.scribe_producer.produce(
                {"payload": json.dumps(command)}, flush=flush, key=oid
            )

    def post_execute(self, result: dict):
        self.produce_scribe(result)
        self.produce_scribe_ztf(result)
        return {}
