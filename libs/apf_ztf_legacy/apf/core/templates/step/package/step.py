from apf.core.step import GenericStep
import logging


class {{step_name}}(GenericStep):
    """{{step_name}} Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """
    def __init__(self,
        config: dict = {},
        level: int = logging.INFO,
        **step_args,
    ):
        super().__init__(config=config, level=level, **step_args)

    def execute(self, message: dict):
        ################################
        #   Here comes the Step Logic  #
        ################################

        pass
