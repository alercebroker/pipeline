from magstats_step.core.factories.object import alerce_object_factory
from magstats_step.core.utils.create_dataframe import *
from magstats_step.core.utils.object_dto import ObjectDTO

def setup_blank_dto(alert):
    detections, non_detections, extra_fields = (
        generic_dataframe_from_detections(alert["detections"]),
        generic_dataframe_from_non_detections(alert["non_detections"]),
        extra_dataframe_from_detections(alert["detections"]),
    )
    alerce_object = alerce_object_factory(alert)
    return ObjectDTO(alerce_object, detections, non_detections, extra_fields)