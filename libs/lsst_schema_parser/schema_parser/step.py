from typing import Any
from apf.core import get_class
from apf.core.step import GenericStep


class SchemaParserStep(GenericStep):
    ingestion_timestamp: int | None

    def __init__(
        self,
        config: dict[str, Any],
        **kwargs: Any,
    ):
        super().__init__(config=config, **kwargs)

        # Get the instance of the parser class
        parser_class = self.config.get("parser_class")
        self.parser = get_class(parser_class)()

    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, messages):
        
        parsed_messages = []
        for message in messages:
            parsed_message = self.parser.parse(message)
            parsed_messages.append(parsed_message)
        
        return parsed_messages  

