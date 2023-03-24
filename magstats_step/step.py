import json
import logging
import pandas as pd

from apf.core import get_class
from apf.core.step import GenericStep
from .core.factories.object import AlerceObject, alerce_object_factory
from .core.utils.magstats_intersection import create_magstats_calculator
from .core.utils.create_dataframe import *


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
        ProducerClass = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = ProducerClass(self.config["SCRIBE_PRODUCER_CONFIG"])

    def object_creator(self, alert):
        return alerce_object_factory(alert)

    def compute_magstats(self, alerts):
        objects = []
        for object_dict in alerts:
            detections = generate_detections_dataframe(object_dict["detections"])
            non_detections = generate_non_detections_dataframe(object_dict["non_detections"])

            alerce_object = AlerceObject(object_dict["aid"])

            alerce_object = self.magstats_calculator(alerce_object, detections, non_detections)[0]
            objects.append(alerce_object.as_dict())

        return objects

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
        print(magstats)
        return magstats
    
    def post_execute(self, result: List[dict]):
        self.produce_scribe(result)
        return result

    def produce_scribe(self, alerce_objects: List[dict]):
        for obj in alerce_objects:
            #TODO: What should be published and where?
            command = {
                "collection": "object",
                "type": "insert",
                "data": obj
            }
            payload = { "payload": json.dumps(command) }
            self.scribe_producer.produce(payload)