import random


class MockMessage:
    def __init__(self, value, topic):
        self.msg_value = value
        self.msg_topic = topic

    def value(self):
        return self.msg_value

    def topic(self):
        return self.msg_topic

    def timestamp(self):
        return (1, 123)


def create_random_string():
    letters = list("abcdefghijklmnopqrstuvwxyz")
    random.shuffle(letters)
    return "".join(letters)


def create_data():
    return {"id": random.getrandbits(32), "message": create_random_string()}


def create_messages(n=1, topic="topic"):
    return [MockMessage(repr(create_data()), topic) for _ in range(n)]
