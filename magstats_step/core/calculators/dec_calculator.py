from ..utils.object_dto import ObjectDTO
from .coordinate_calculator import calculate_stats_coordinates


def calculate_dec(object_dto: ObjectDTO) -> ObjectDTO:
    dec_series, e_dec_series = object_dto.detections["dec"], object_dto.detections["e_dec"]
    meandec, dec_error = calculate_stats_coordinates(dec_series, e_dec_series)

    populated_object = object_dto.alerce_object.copy()
    populated_object["meandec"] = meandec
    populated_object["sigmadec"] = dec_error

    return ObjectDTO(
        populated_object,
        object_dto.detections,
        object_dto.non_detections,
        object_dto.extra_fields,
    )
