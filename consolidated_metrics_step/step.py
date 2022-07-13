from .utils.metric import ConsolidatedMetric
from .utils.metric import STEP_MAPPER
from .utils.metric import StepMetric
from apf.core.step import GenericStep
from datetime import datetime

import logging


class ConsolidatedMetricsStep(GenericStep):
    """ConsolidatedMetricsStep Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(self, consumer=None, config=None, level=logging.INFO, **step_args):
        super().__init__(consumer, config=config, level=level)
        self.datetime_conversion = lambda x: datetime.strptime(
            x, "%Y-%m-%dT%H:%M:%S.%f+00:00"
        )
        self.pipeline_order = config.get("PIPELINE_ORDER")

    @staticmethod
    def generate_consolidated_metrics(
        candid: str, source: str, metric: StepMetric
    ) -> ConsolidatedMetric:
        query = ConsolidatedMetric.find(ConsolidatedMetric.candid == candid).all()
        if len(query):  # HIT
            consolidated_metric = query[0]
            consolidated_metric[source] = metric
        else:  # MISS
            kwargs = {"candid": candid, source: metric}
            consolidated_metric = ConsolidatedMetric(**kwargs)
        consolidated_metric.save()

        print(consolidated_metric)

        if consolidated_metric.is_bingo():
            print(consolidated_metric)
        return consolidated_metric

    def execute(self, message):
        ################################
        #   Here comes the Step Logic  #
        ################################

        for msg in message:
            print(msg)
            if "candid" not in msg.keys():
                return
            candid = msg["candid"]
            source = STEP_MAPPER[msg["source"]]
            metric = StepMetric(
                received=self.datetime_conversion(msg["timestamp_received"]),
                sent=self.datetime_conversion(msg["timestamp_sent"]),
                execution_time=msg["execution_time"],
            )

            if isinstance(candid, list):
                for c in candid:
                    cm = self.generate_consolidated_metrics(c, source, metric)
            else:
                cm = self.generate_consolidated_metrics(candid, source, metric)
