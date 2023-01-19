from json import loads
from .commands import DbCommand
from .exceptions import MisformattedCommandExcepction


def validate(message: dict):
    if "type" not in message or "data" not in message or "collection" not in message:
        return None

    if "criteria" not in message:
        message["criteria"] = None

    return message


def decode_message(encoded_message: str):
    decoded = loads(encoded_message)
    valid_message = validate(decoded)

    if valid_message is None:
        raise MisformattedCommandExcepction

    return valid_message


def db_command_factory(msg: str):
    decoded_message = decode_message(msg)
    return DbCommand(
        decoded_message["collection"],
        decoded_message["type"],
        decoded_message["criteria"],
        decoded_message["data"],
    )
