import logging

from apf.core.step import GenericStep

from magstats_step.core.objstats import ObjectStatistics


class MagstatsStep(GenericStep):
    def __init__(
        self,
        config={},
        level=logging.INFO,
        **step_args,
    ):
        super().__init__(config=config, level=level, **step_args)
        self.excluded = set(config["EXCLUDED_CALCULATORS"])

    def compute_magstats(self, alerts):
        outputs = []
        for alert in alerts:
            calculator = ObjectStatistics(**alert, exclude=self.excluded)
            outputs.append(calculator.generate_object())
        return outputs

    def execute(self, messages: list):
        self.logger.info(f"Processing {len(messages)} alerts")
        magstats = self.compute_magstats(messages)
        self.logger.info(f"Clean batch of data\n")
        return magstats
