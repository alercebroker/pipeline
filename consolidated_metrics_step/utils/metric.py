from datetime import datetime
from redis_om import EmbeddedJsonModel
from redis_om import Field
from redis_om import JsonModel
from typing import Optional

STEP_MAPPER = {
    "S3Step": "s3",
    "EarlyClassifier": "early_classifier",
    "WatchlistStep": "watchlist",
    "SortingHatStep": "sorting_hat",
    "IngestionStep": "ingestion",
    "XmatchStep": "xmatch",
    "FeaturesComputer": "features",
    "LateClassifier": "late_classifier",
}


class StepMetric(EmbeddedJsonModel):
    received: datetime
    sent: datetime
    execution_time: float


class ConsolidatedMetric(JsonModel):
    candid: str = Field(index=True)
    s3: Optional[StepMetric]
    early_classifier: Optional[StepMetric]
    watchlist: Optional[StepMetric]
    sorting_hat: Optional[StepMetric]
    ingestion: Optional[StepMetric]
    xmatch: Optional[StepMetric]
    features: Optional[StepMetric]
    late_classifier: Optional[StepMetric]

    def __getitem__(self, field):
        return self.__dict__["__field__"][field]

    def __setitem__(self, key, value):
        pass


from redis_om import get_redis_connection
from redis_om import Migrator

redis = get_redis_connection()
Migrator().run()
