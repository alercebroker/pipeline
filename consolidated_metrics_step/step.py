from .utils.metric import ConsolidatedMetric
from .utils.metric import STEP_MAPPER
from .utils.metric import StepMetric
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

    def get_survey(self, candid: str) -> str:
        if candid[:3] in ["01a", "02a", "03a", "04a"]:
            return "ATLAS"
        elif len(candid) == 19:
            return "ZTF"
        return "UNKNOWN"

    def generate_consolidated_metrics(
        self, candid: str, source: str, metric: StepMetric
    ) -> ConsolidatedMetric:
        query = ConsolidatedMetric.find(ConsolidatedMetric.candid == candid).all()
        if len(query):  # HIT
            consolidated_metric = query[0]
            consolidated_metric[source] = metric
        else:  # MISS
            survey = self.get_survey(candid)
            kwargs = {"candid": candid, "survey": survey, source: metric}
            consolidated_metric = ConsolidatedMetric(**kwargs)
            consolidated_metric.expire(21600)  # 21600 seconds -> 6 hours
        consolidated_metric.save()

        survey = consolidated_metric.survey
        pipeline = self.pipeline_order[survey]

        self.logger.debug(f"Candid {candid} is {consolidated_metric.status(pipeline)}")

        if consolidated_metric.is_bingo(pipeline):
            queue_times = consolidated_metric.compute_queue_times(
                self.pipeline_order[survey]
            )
            distance = self.pipeline_distances[survey]
            total_time_in_pipeline = consolidated_metric.compute_total_time(*distance)
            execution_time_in_pipeline = consolidated_metric.compute_execution_time(
                self.pipeline_order[survey]
            )
            output = {
                "candid": candid,
                "total_time": total_time_in_pipeline,
                "execution_time": execution_time_in_pipeline,
                "survey": survey,
                **queue_times,
            }
            self.producer.send_metrics(output)
            ConsolidatedMetric.delete(consolidated_metric.pk)
            self.producer.producer.poll(0.0)  # Do poll for sync the production
            self.logger.info(f"Produced consolidated metrics for candid: {candid}")
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
                received=dateutil.parser.parse(msg["timestamp_received"]),
                sent=dateutil.parser.parse(msg["timestamp_sent"]),
                execution_time=msg["execution_time"],
            )

            if isinstance(candid, list):
                for c in candid:
                    cm = self.generate_consolidated_metrics(c, source, metric)
            else:
                cm = self.generate_consolidated_metrics(candid, source, metric)
        return
