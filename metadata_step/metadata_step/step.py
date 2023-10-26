from typing import Any, Dict, Iterable, List, Union

from apf.core.step import GenericStep, get_class


class MetadataStep(GenericStep):
    def __init__(self, config, **step_args):
        super().__init__(config=config, **step_args)
        cls = get_class(config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(config["SCRIBE_PRODUCER_CONFIG"])

    def execute(self, messages: List[Dict]):
        for message in messages:
            print(message)

    def post_execute(self, result: List[Dict]):
        pass