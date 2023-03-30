import logging
from typing import List

from apf.core.step import GenericStep

from magstats_step.core import ObjectStatistics, MagnitudeStatistics


class MagstatsStep(GenericStep):
    def __init__(
        self,
        config,
        level=logging.INFO,
        **step_args,
    ):
        super().__init__(config=config, level=level, **step_args)
        self.excluded = set(config["EXCLUDED_CALCULATORS"])

    @classmethod
    def pre_execute(cls, messages: List[dict]) -> dict:
        detections, non_detections = [], []
        for msg in messages:
            detections.extend(msg["detections"])
            non_detections.extend(msg["non_detections"])
        return {"detections": detections, "non_detections": non_detections}

    def execute(self, messages: dict):
        obj_calculator = ObjectStatistics(messages["detections"])
        stats = obj_calculator.generate_statistics(self.excluded).to_dict("index")

        magstats_calculator = MagnitudeStatistics(**messages)
        magstats = magstats_calculator.generate_statistics(self.excluded).reset_index()
        magstats = magstats.set_index(magstats["aid"] + magstats["fid"].astype(str)).to_dict("index")

        return {"objects": stats, "magstats": magstats}
