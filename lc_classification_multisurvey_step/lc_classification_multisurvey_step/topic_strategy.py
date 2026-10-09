"""Daily topic strategy for the output producer.

apf's DailyTopicStrategy only moves to the next day's topic if get_topics() runs between
change_hour and midnight UTC (and then keeps the old day in the list for a while). A
producer only calls it when it produces, and ZTF has no traffic at 23 UTC, so a pod that
lives across the day change keeps writing to the topic of the day it started.

This strategy computes the topic from the clock on every call and returns only the
current day's topic(s). Producer only: a consumer using it would drop yesterday's topics
at the day change.
"""

import datetime

from apf.core.topic_management import DailyTopicStrategy


class CurrentDailyTopicStrategy(DailyTopicStrategy):
    def get_topics(self):
        now = datetime.datetime.utcnow()
        if now.hour >= self.change_hour:
            now += datetime.timedelta(days=1)
        date = now.strftime(self.date_format)
        return [topic_format % date for topic_format in self.topic_formats]
