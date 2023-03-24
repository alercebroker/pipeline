import logging

from apf.core.step import GenericStep

from .core.calculators.object import ObjectStatistics


class MagstatsStep(GenericStep):
    """MagstatsStep Description
    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)
    """

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
            calculator = ObjectStatistics(**alert)
            outputs.append(calculator.generate_object(self.excluded))
        return outputs

    def execute(self, messages: list):
        """TODO: Docstring for execute.
        TODO:

        :messages: TODO
        :returns: TODO

        """
        self.logger.info(f"Processing {len(messages)} alerts")
        magstats = self.compute_magstats(messages)
        self.logger.info(f"Clean batch of data\n")
        return magstats
