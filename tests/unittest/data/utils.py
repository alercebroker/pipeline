from magstats_step.core.factories.object import alerce_object_factory
from magstats_step.core.utils.create_dataframe import *

def setup_blank_dto(alert):
    detections, non_detections = (
        generate_detections_dataframe(alert["detections"]),
        generate_non_detections_dataframe(alert["non_detections"]),
    )
    alerce_object = alerce_object_factory(alert)
    return alerce_object, detections, non_detections