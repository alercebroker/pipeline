from datetime import datetime
from redis_om import EmbeddedJsonModel, Field, JsonModel
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
    execution_time: float  # in seconds


class ConsolidatedMetric(JsonModel):
    candid: str = Field(index=True)
    survey: str
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
        if key in STEP_MAPPER.keys():
            key = STEP_MAPPER[key]
            self.__setattr__(key, value)
        else:
            self.__setattr__(key, value)

    def _get_mapped_attribute(self, key):
        return self.__getattribute__(STEP_MAPPER[key])

    def _get_prev_step(self, pipeline_order: dict, to_find: str) -> [str, None]:
        for head, tail in pipeline_order.items():
            if tail:
                for h, t in tail.items():
                    if h == to_find:
                        return head
                    else:
                        return self._get_prev_step(tail, to_find)
        return None

    def compute_queue_time_between(
        self, prev_step: str, current_step: str
    ) -> [float, None]:
        t_0 = self._get_mapped_attribute(prev_step)
        t_1 = self._get_mapped_attribute(current_step)
        queue_time = None
        if t_0 and t_1:
            queue_time = t_1.received.replace(tzinfo=None) - t_0.sent.replace(
                tzinfo=None
            )
            queue_time = queue_time.total_seconds()
        return queue_time

    def compute_queue_time(
        self, pipeline_order: dict, current_step: str
    ) -> [float, None]:
        prev_step = self._get_prev_step(pipeline_order, current_step)
        if prev_step:
            queue_time = self.compute_queue_time_between(prev_step, current_step)
            return queue_time
        return None

    def compute_accumulated_time(
        self, pipeline_order: dict, current_step: str
    ) -> [float, None]:
        prev = self._get_prev_step(pipeline_order, current_step)
        t_1 = self._get_mapped_attribute(current_step)
        accumulated_time = t_1.execution_time
        while prev:
            queue_time = self.compute_queue_time(pipeline_order, current_step)
            t_0 = self._get_mapped_attribute(prev)
            if t_0 is None:
                return None
            accumulated_time += t_0.execution_time + queue_time
            current_step = prev
            prev = self._get_prev_step(pipeline_order, current_step)
        return accumulated_time

    @staticmethod
    def get_deep_pipeline(pipeline_order: dict):
        def get_deep(sender, receiver, response=None):
            if response is None:
                response = []

            response.append(sender)
            if receiver is None:
                return
            for k, v in receiver.items():
                get_deep(k, v, response)

        r = []
        for head, tail in pipeline_order.items():
            get_deep(head, tail, r)

        return [STEP_MAPPER[x] for x in r]

    def status(self, pipeline_order: dict) -> str:
        keys = self.get_deep_pipeline(pipeline_order)
        res = 0
        for k in keys:
            if self.__getattribute__(k):
                res += 1
        return f"{res}/{len(keys)}"

    def is_bingo(self, pipeline_order: dict) -> bool:
        keys = self.get_deep_pipeline(pipeline_order)
        for k in keys:
            if self.__getattribute__(k) is None:
                return False
        return True


# from redis_om import get_redis_connection
# from redis_om import Migrator
#
# redis = get_redis_connection()
# Migrator().run()
