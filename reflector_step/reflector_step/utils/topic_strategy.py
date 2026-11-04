import datetime

from apf.core.topic_management import DailyTopicStrategy


class RollingDailyTopicStrategy(DailyTopicStrategy):
    """
    Consumer strategy: the current day's topics plus the `previous_days` days before it,
    computed from the UTC clock on every call.

    apf's DailyTopicStrategy builds the list from the moment the pod started: a pod started
    after the day change only gets today's topics, so what was left in yesterday's topics is
    orphaned when the older pods die. Here every pod gets the same list no matter when it
    started, so any live pod drains the previous days, and the oldest day leaves the list on
    its own at change_hour.

    CONSUMER ONLY: an apf producer writes every message to every topic of the strategy.
    """

    def __init__(self, previous_days=1, **kwargs):
        super().__init__(**kwargs)
        if previous_days < 0:
            raise ValueError("previous_days must be >= 0")
        self.previous_days = previous_days

    def get_topics(self):
        current = datetime.datetime.utcnow()
        if current.hour >= self.change_hour:
            current += datetime.timedelta(days=1)
        topics = []
        for days_back in range(self.previous_days, -1, -1):
            date = current - datetime.timedelta(days=days_back)
            topics += [
                topic_format % date.strftime(self.date_format)
                for topic_format in self.topic_formats
            ]
        return topics
