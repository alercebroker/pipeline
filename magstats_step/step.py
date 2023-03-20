from apf.core.step import GenericStep
import pandas as pd
import logging

from .strategies.magstats_computer import MagstatsComputer
from .strategies.ztf_strategy import ZTFMagstatsStrategy
from .core.factories.object import alerce_object_factory

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
        self.magstats_computer = MagstatsComputer(ZTFMagstatsStrategy())
    
    def object_creator(self, alert):
        return alerce_object_factory(alert) 

    def compute_magstats(self, alerts):
        magstats_list = []
        for object_dict in alerts:
            detections = object_dict["detections"]
            non_detections = object_dict["non_detections"]
            #TODO: Check if object is atlas or ztf
            self.magstats_computer.strategy = ZTFMagstatsStrategy()
            magstats_dict = self.magstats_computer.compute_magstats(detections, non_detections)
            response_item = {
                    'aid' : object_dict['aid'],
                    'magstats' : magstats_dict
                             }
            magstats_list.append(response_item)
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
