import os
import sys

from apf.core.step import GenericStep
from alerce_classifiers.transformer_online_classifier import TransformerOnlineClassifier
import logging


class TransformerOnlineClassifierStep(GenericStep):
    """TransformerOnlineClassifierStep Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """
    def __init__(self, consumer=None, config=None, level=logging.INFO, **step_args):
        super().__init__(consumer, config=config, level=level)
        self.modelo = TransformerOnlineClassifier("../Encoder.pt")

    def execute(self, message):
        print(message)
        # self.modelo.predict_proba(message)
        input("stop")
