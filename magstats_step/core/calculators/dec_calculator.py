from ..utils.object_dto import ObjectDTO
from .coordinate_calculator import calculate_stats_coordinates


def calculate_dec(object_dto: ObjectDTO) -> ObjectDTO:
    dec_list = [det["dec"] for det in object_dto.detections]
    e_dec_list = [det["e_dec"] for det in object_dto.detections]
    meandec, dec_error = calculate_stats_coordinates(dec_list, e_dec_list)

    populated_object = object_dto.alerce_object.copy()
    populated_object["meandec"] = meandec
    populated_object["sigmadec"] = dec_error

    return ObjectDTO(
        populated_object,
        object_dto.detections,
        object_dto.non_detections
    )
