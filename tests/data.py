from datetime import datetime
from datetime import timedelta
from faker import Faker
from faker import providers


class FakeMetric:
    def __init__(self):
        self.sources = [
            "S3Step",
            "EarlyClassifier",
            "WatchlistStep",
            "SortingHatStep",
            "IngestionStep",
            "XmatchStep",
            "FeaturesComputer",
            "LateClassifier",
        ]
        self.fake = Faker()
        self._init_faker()

    def _init_faker(self):
        sources_provider = providers.DynamicProvider(
            provider_name="sources",
            elements=self.sources,
        )
        self.fake.add_provider(sources_provider)

    def create_fake_metric(self, candid: str = None, source: str = None):
        fake_candid = self.fake.user_name()
        fake_candids = [self.fake.user_name() for _ in range(0, 10)]
        return {
            "timestamp_sent": self.fake.date_time_between(
                start_date=datetime.now() - timedelta(minutes=10),
                end_date=datetime.now(),
            ).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"),
            "timestamp_received": self.fake.date_time_between(
                start_date=datetime.now() - timedelta(minutes=30),
                end_date=datetime.now() - timedelta(minutes=20),
            ).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"),
            "execution_time": self.fake.random.uniform(0, 1),
            "candid": candid or self.fake.random.choice([fake_candids, fake_candid]),
            "source": source or self.fake.sources(),
        }

    def create_fake_metrics_candid(self, candid: str):
        return [self.create_fake_metric(candid, s) for s in self.sources]
