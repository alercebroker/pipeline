from apf.consumers.generic import GenericConsumer
import json


class TestConsumer(GenericConsumer):
    def __init__(self, config: dict, offset=0):
        super().__init__(config)
        messages_path = config["PARAMS"]["input_path"]
        messages_format = config["PARAMS"]["input_format"]
        if messages_format == "json":
            with open(messages_path, 'r') as f:
                list_messages = json.load(f)
                
        # if type_file == 'avro':  ## Se pueden añadir más formatos siempre en cuando el input finalmente sea una lista de dicts?
        #     avro.load(path_file)
        # If there's not list messages, it means it wasn't possible to load the input, so an error should be raised
        if not list_messages:
            raise Exception("Error loading messages from file")

        self.config = config
        self.messages = list_messages
        self.logger.info(
            f"Creating test consumer"
        )

    def consume(self): # Returning all the messages without using offset
        messages_to_return = self.messages
        self.logger.info(
            f"Consumed {len(messages_to_return)} messages."
        )
        yield messages_to_return

    def commit(self):
        self.logger.info(
            f"We have consumed {len(self.messages)} messages. These are all the messages available for this consumer."
        )
        return True
