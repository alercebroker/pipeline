import datetime

class GenericTopicStrategy:
    def get_topic(self):
        pass


class DailyTopicStrategy(GenericTopicStrategy):
    """ Gives a KafkaConsumer a new topic every day ().

    Parameters
    ----------
    topic_format : str/list
        Topic format with %s for the date field.
    date_format : str
        Date formart string i.e. "%Y%m%d".
    change_hour: int
        UTC hour to change the topic.

    """
    def __init__(self,topic_format="ztf_%s_programid1",date_format="%Y%m%d", change_hour = 22):
        super().__init__()
        self.topic_format = topic_format if type(topic_format) is list else [topic_format]
        self.date_format  = date_format
        self.change_hour  = change_hour

    def get_topic(self):
        """Get list of topics updated to the current date.

        Returns
        -------
        :class:`list`
            List of updated topics.

        """
        now = datetime.datetime.utcnow()
        if now.hour >= self.change_hour:
            now += datetime.timedelta(days=1)
        date = now.strftime(self.date_format)
        topics = [topic % date for topic in self.topic_format]
        return topics
