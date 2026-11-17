"""Send a list of scribe commands to the scribe topic."""
import json  # encode each command


def send_commands(producer, topic: str, commands: list) -> None:
    """Produce every command to `topic` in order, keyed by its payload oid, then flush once.

    An empty list produces nothing and does not flush.
    """
    for command in commands:  # in list order
        oid = command["payload"]["oid"]  # object id of the command
        producer.produce(  # send the command
            topic=topic,  # scribe topic
            value=json.dumps(command).encode("utf-8"),  # command as json bytes
            key=str(oid).encode("utf-8"),  # keyed by oid
        )  # end of the produce call
    if commands:  # something was produced
        producer.flush()  # deliver the batch once, after the last command
