from apf.core.step import GenericStep
import pandas as pd
import logging

from .core.factories.object import alerce_object_factory
from .core.utils.magstats_intersection import create_magstats_calculator
from .core.utils.object_dto import ObjectDTO


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
        self.magstats_calculator = create_magstats_calculator(
            config["EXCLUDED_CALCULATORS"]
        )

    def object_creator(self, alert):
        return alerce_object_factory(alert)

    ## IMPORTANT THING TO NOTICE: The detections (and non detections)
    ## must be sorted by mjd in order to work properly
    def compute_magstats(self, alerts):
        magstats_list = []
        for object_dict in alerts:
            detections = list(object_dict["detections"]).sort(
                key=lambda det: det["mjd"]
            )
            non_detections = list(object_dict["non_detections"]).sort(
                key=lambda nondet: nondet["mjd"]
            )
            # TODO: Check if object is atlas or ztf
            object_dto = ObjectDTO(
                self.object_creator(alerts), detections, non_detections
            )
            object_with_magstats = self.magstats_calculator(object_dto)
            magstats_list.append(object_with_magstats.alerce_object)

        return magstats_list

    def execute(self, messages: list):
        """TODO: Docstring for execute.
        TODO:

        :messages: TODO
        :returns: TODO

        """
        self.logger.info(f"Processing {len(messages)} alerts")

        magstats = self.compute_magstats(messages)

        # self.magstats_calculator.insert(new_stats, self.driver)
        self.logger.info(f"Clean batch of data\n")
        return magstats
