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

    def _get_mapped_attribute(self, key):
        return self.__getattribute__(STEP_MAPPER[key])

    def is_bingo(self) -> bool:
        keys = STEP_MAPPER.values()
        for k in keys:
            if self.__getattribute__(k) is None:
                return False
        return True

    def compute_queue_times(self, pipeline_order: dict) -> dict:
        def compute_queue(sender, receiver, response=None):
            if response is None:
                response = {}
            for k, v in receiver.items():
                queue_time = (
                    self._get_mapped_attribute(k).received
                    - self._get_mapped_attribute(sender).sent
                )
                response[f"{sender}_{k}"] = queue_time.total_seconds()
                if v is not None:
                    compute_queue(k, v, response)

        if not self.is_bingo():
            missing = [
                k for k in STEP_MAPPER.values() if self.__getattribute__(k) is None
            ]
            raise Exception(
                f"Consolidated metric is not full yet (bad bingo): Missing metrics for {missing}"
            )

        queue_times = {}
        for head, tail in pipeline_order.items():
            if tail is not None:
                compute_queue(head, tail, queue_times)
        return queue_times

    def compute_total_time(self, start: str, end: str) -> float:
        total_time = (
            self.__getattribute__(end).sent - self.__getattribute__(start).received
        )
        return total_time.total_seconds()

    def __getitem__(self, field):
        return self.__dict__["__field__"][field]

    def __setitem__(self, key, value):
        self.__setattr__(key, value)


# from redis_om import get_redis_connection
# from redis_om import Migrator
#
# redis = get_redis_connection()
# Migrator().run()
