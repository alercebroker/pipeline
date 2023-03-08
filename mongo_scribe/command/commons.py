from collections import namedtuple


_Commands = namedtuple("Commands", ("insert", "update", "update_probabilities"))

ValidCommands = _Commands("insert", "update", "update_probabilities")
