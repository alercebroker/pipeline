from apf.core.step import GenericStep
import logging


class CustomMirrormaker(GenericStep):
    """CustomMirrormaker Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    config : dict
        Configuration dictionary.
    level : int
        Logging level.
    """

    def __init__(self, consumer=None, config=None, level=logging.INFO):
        super().__init__(consumer, config=config, level=level)

    def execute(self, message):
        pass
