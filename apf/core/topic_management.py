import datetime


class GenericTopicStrategy:
    def get_topics(self):
        pass


class Topic:
    def __init__(self, name, date, name_format, date_format):
        self.name = name
        self.date = date
        self.name_format = name_format
        self.date_format = date_format

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class DailyTopicStrategy(GenericTopicStrategy):
    """Gives a KafkaConsumer a new topic every day. (For more information check the :class:`apf.consumers.KafkaConsumer`)

    Parameters
    ----------
    topic_format : str/list
        Topic format with %s for the date field.
    date_format : str
        Date format code. Must be a valid 1989 C code.
            i.e: %Y%m%d = YYYY-mm-dd
    change_hour: int
        UTC hour to change the topic.

    """

    def __init__(
        self,
        topic_format="ztf_%s_programid1",
        date_format="%Y%m%d",
        change_hour=22,
        retention_days=8,
    ):
        super().__init__()
        self.topic_formats = (
            topic_format if type(topic_format) is list else [topic_format]
        )
        self.retention_days = retention_days
        self.date_format = date_format
        self.change_hour = change_hour
        now = datetime.datetime.utcnow()
        date = now.strftime(self.date_format)
        self.topics = [
            Topic(topic % date, now, topic, self.date_format)
            for topic in self.topic_formats
        ]

    def _remove_old_topics(self, now):
        for topic in self.topics:
            delta = now - topic.date
            if abs(delta.days) >= self.retention_days:
                self.topics = self.topics[-self.retention_days :]

    def get_topics(self):
        """Get list of topics updated to the current date.

        Returns
        -------
        :class:`list`
            List of updated topics.

        """
        now = datetime.datetime.utcnow()
        self._remove_old_topics(now)
        if now.hour >= self.change_hour:
            now += datetime.timedelta(days=1)
            date = now.strftime(self.date_format)
            for topic_format in self.topic_formats:
                topic = Topic(topic_format % date, now, topic_format, self.date_format)
                if not [x for x in self.topics if x.name == topic.name]:
                    self.topics.append(topic)

        return [topic.name for topic in self.topics]
