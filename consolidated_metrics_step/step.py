from .utils.metric import ConsolidatedMetric
from .utils.metric import STEP_MAPPER
from .utils.metric import StepMetric
from apf.core.step import GenericStep
from apf.metrics import KafkaMetricsProducer
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
        self.producer = KafkaMetricsProducer(self.config["PRODUCER_CONFIG"])

    def generate_consolidated_metrics(
        self, candid: str, source: str, metric: StepMetric
    ) -> ConsolidatedMetric:
        query = ConsolidatedMetric.find(ConsolidatedMetric.candid == candid).all()
        if len(query):  # HIT
            consolidated_metric = query[0]
            consolidated_metric[source] = metric
        else:  # MISS
            kwargs = {"candid": candid, source: metric}
            consolidated_metric = ConsolidatedMetric(**kwargs)
        consolidated_metric.save()

        if consolidated_metric.is_bingo():
            queue_times = consolidated_metric.compute_queue_times(self.pipeline_order)
            total_time_in_pipeline = consolidated_metric.compute_total_time(
                "sorting_hat", "late_classifier"
            )
            output = {
                "candid": candid,
                "total_time": total_time_in_pipeline,
                **queue_times,
            }
            self.producer.send_metrics(output)
            self.producer.producer.poll(0.0)  # Do poll for sync the production
            self.logger.info(f"Produced consolidated metrics for: {candid}")
        return consolidated_metric

    def execute(self, message):
        ################################
        #   Here comes the Step Logic  #
        ################################

        for msg in message:
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

        return
