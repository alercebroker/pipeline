from .utils.metric import ConsolidatedMetric, STEP_MAPPER, StepMetric
from apf.core.step import GenericStep
from apf.metrics import KafkaMetricsProducer

import dateutil.parser
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
        self.pipeline_order = config.get("PIPELINE_ORDER")
        self.pipeline_distances = config.get("PIPELINE_DISTANCES")
        self.producer = KafkaMetricsProducer(self.config["PRODUCER_CONFIG"])
        self.expire_time = config.get("EXPIRE_TIME", 21600)  # 21600 seconds -> 6 hours

    def produce(self, message: dict) -> None:
        self.producer.send_metrics(message)
        self.producer.producer.poll(0.0)  # Do poll for sync the production
        self.logger.info(f"Produced queue metrics for candid: {message['candid']}")

    def process_queues(
        self, consolidated_metric: ConsolidatedMetric, last_source: str
    ) -> [dict, None]:
        candid = consolidated_metric.candid
        survey = consolidated_metric.survey
        pipeline = self.pipeline_order[survey]
        milestone = last_source

        self.logger.info(
            f"Candid {candid} is {consolidated_metric.status(pipeline)} in {milestone}"
        )

        queue_time = consolidated_metric.compute_queue_time(pipeline, last_source)
        accumulated_time = consolidated_metric.compute_accumulated_time(
            pipeline, last_source
        )

        # The metrics are right. It means that all metrics to past is completed.
        if accumulated_time and queue_time:
            partial_metric = {
                "candid": candid,
                "survey": survey,
                "milestone": f"Queue_{milestone}",
                "queue_time": queue_time,
                "accumulated_time": accumulated_time,
            }
            return partial_metric
        return None

    def generate_consolidated_metrics(
        self, candid: str, source: str, metric: StepMetric, survey: str
    ) -> ConsolidatedMetric:
        query = ConsolidatedMetric.find(ConsolidatedMetric.candid == candid).all()
        source = STEP_MAPPER[source]
        if len(query):  # HIT
            consolidated_metric = query[0]
            consolidated_metric[source] = metric
        else:  # MISS
            kwargs = {"candid": candid, "survey": survey, source: metric}
            consolidated_metric = ConsolidatedMetric(**kwargs)
        consolidated_metric.save()
        consolidated_metric.expire(self.expire_time)

        return consolidated_metric

    def process_metric(self, candid: str, survey: str, source: str, metric: StepMetric):
        pipeline = self.pipeline_order[survey]
        cm = self.generate_consolidated_metrics(candid, source, metric, survey)
        metric_queue = self.process_queues(cm, source)
        self.produce(metric_queue)
        if cm.is_bingo(pipeline):
            ConsolidatedMetric.delete(cm.pk)

    def execute(self, message):
        ################################
        #   Here comes the Step Logic  #
        ################################
        for msg in message:
            if "candid" not in msg.keys():
                return
            candid = msg["candid"]
            survey = msg["tid"]
            source = msg["source"]

            metric = StepMetric(
                received=dateutil.parser.parse(msg["timestamp_received"]),
                sent=dateutil.parser.parse(msg["timestamp_sent"]),
                execution_time=msg["execution_time"],
            )
            if isinstance(candid, list):
                for c, s in zip(candid, survey):
                    self.process_metric(c, s, source, metric)

            else:
                self.process_metric(candid, survey, source, metric)
        return
