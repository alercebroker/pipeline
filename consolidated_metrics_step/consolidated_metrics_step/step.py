from .utils.metric import ConsolidatedMetric
from .utils.metric import StepMetric
from .utils.metric import STEP_MAPPER
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

    def __init__(
        self, consumer=None, config=None, level=logging.INFO, producer=None, **step_args
    ):
        super().__init__(consumer, config=config, level=level)
        self.pipeline_order = config.get("PIPELINE_ORDER")
        self.producer = producer
        if config.get("PRODUCER_CONFIG", False):
            self.producer = KafkaMetricsProducer(self.config["PRODUCER_CONFIG"])
        self.expire_time = config.get("EXPIRE_TIME", 21600)  # 21600 seconds -> 6 hours

    @staticmethod
    def retrieve_survey(candid: str, survey: [str, None]) -> str:
        if survey:
            return survey
        elif len(candid) == 19 and candid.isdigit():
            return "ZTF"
        elif candid[:3] in ["01a", "02a", "03a", "04a"]:
            tid = candid[:3]  # the identifier of ATLAS
            return f"ATLAS-{tid}"
        else:
            return "UNKNOWN"

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
        try:
            query = ConsolidatedMetric.find(ConsolidatedMetric.candid == candid)
            query = query.all()
        except TypeError as e:
            self.logger.info(
                f"{candid} of {source}-{survey}\n{e}"
            )  # Sometimes raise an error of redis_om, found a None response after that try to indexing as str
            query = []

        source = STEP_MAPPER[source]
        if len(query):  # HIT
            consolidated_metric = query[0]
            consolidated_metric[source] = metric
            if consolidated_metric.survey is None:
                consolidated_metric.survey = survey
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
        if metric_queue:
            self.produce(metric_queue)
            if cm.is_bingo(pipeline):
                ConsolidatedMetric.delete(cm.pk)

    def execute(self, message):
        ################################
        #   Here comes the Step Logic  #
        ################################
        for msg in message:
            candid = msg.get("candid", None)
            if candid is None:
                continue
            source = msg["source"]
            tid = msg.get("tid", None)
            survey = self.retrieve_survey(candid, tid)

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
