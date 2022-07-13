from faker import Faker, providers
import datetime


class FakeMetric:
    def __init__(self):
        self.fake = Faker()
        self._init_faker()

    def _init_faker(self):
        sources_provider = providers.DynamicProvider(
            provider_name="sources",
            elements=['S3Step', 'EarlyClassifier', 'WatchlistStep', 'SortingHatStep', 'IngestionStep', 'XmatchStep',
                      'FeaturesComputer', 'LateClassifier'],
        )
        self.fake.add_provider(sources_provider)

    def create_fake_metric(self, candid: str = None, source: str = None):
        fake_candid = self.fake.user_name()
        fake_candids = [self.fake.user_name() for x in range(0, 10)]
        return {
            "timestamp_sent": self.fake.date(pattern="%Y-%m-%dT%H:%M:%S.%f+00:00",
                                             end_datetime=datetime.datetime.now()),
            "timestamp_received": self.fake.date(pattern="%Y-%m-%dT%H:%M:%S.%f+00:00",
                                                 end_datetime=datetime.datetime.now()),
            "execution_time": self.fake.random.uniform(0, 1),
            "candid": candid or self.fake.random.choice([fake_candids, fake_candid]),
            "source": source or self.fake.sources()
        }
