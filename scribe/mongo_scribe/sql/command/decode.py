from json import loads
from .exceptions import WrongFormatCommandException
from .commands import (
    Command,
    InsertObjectCommand,
    InsertDetectionsCommand,
    InsertForcedPhotometryCommand,
    UpdateObjectStatsCommand,
    UpsertFeaturesCommand,
    UpsertNonDetectionsCommand,
    UpsertProbabilitiesCommand,
    UpsertXmatchCommand,
)


def validate(message: dict) -> dict:
    """Checks if a dictionary has a valid command format. Returns the dictionary if valid.

    It raises MisformattedCommand if the JSON string isn't a valid command.
    """
    if any(required not in message for required in ["type", "data", "collection"]):
        raise WrongFormatCommandException

    if "criteria" not in message:
        message["criteria"] = {}

    if "options" not in message:
        message["options"] = {}

    # Removing multistream keys
    if "aid" in message["criteria"]:
        message["criteria"].pop("aid")

    if "_id" in message["criteria"]:
        message["criteria"].pop("_id")

    return message


def decode_message(encoded_message: str) -> dict:
    """
    Get the JSON string and transforms it into a Python dictionary.
    """
    decoded = loads(encoded_message)
    valid_message = validate(decoded)
    return valid_message


def command_factory(msg: str) -> Command:
    message = decode_message(msg)
    type_ = message.pop("type")
    table = message.pop("collection")

    # here it comes
    if type_ == "insert" and table == "object":
        return InsertObjectCommand(**message)
    if type_ == "update" and table == "detection":
        return InsertDetectionsCommand(**message)
    if type_ == "upsert" and table == "magstats":
        return UpdateObjectStatsCommand(**message)
    if type_ == "update" and table == "non_detection":
        return UpsertNonDetectionsCommand(**message)
    if type_ == "update_features" and table == "object":
        return UpsertFeaturesCommand(**message)
    if type_ == "update_probabilities" and table == "object":
        return UpsertProbabilitiesCommand(**message)
    if type_ == "update" and table == "object" and "xmatch" in message["data"]:
        return UpsertXmatchCommand(**message)
    if type_ == "update" and table == "forced_photometry":
        return InsertForcedPhotometryCommand(**message)
    raise ValueError(f"Unrecognized command type {type_} in table {table}.")
