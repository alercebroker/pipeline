from typing import Any
from apf.core.step import GenericStep


class SchemaParser(GenericStep):
    ingestion_timestamp: int | None

    def __init__(
        self,
        config: dict[str, Any],
        **kwargs: Any,
    ):
        super().__init__(config=config, **kwargs)

    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, messages: list[Message]
    ) -> ParsedData:
        pass

    def pre_produce(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, result: ParsedData
    ):
        self.set_producer_key_field(self.Strategy.get_key())
        messages = self.Strategy.serialize(result)

        return messages
