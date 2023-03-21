from ..utils.object_dto import ObjectDTO
from .coordinate_calculator import calculate_stats_coordinates


def calculate_ra(object_dto: ObjectDTO) -> ObjectDTO:
    ra_list = [det["ra"] for det in object_dto.detections]
    e_ra_list = [det["e_ra"] for det in object_dto.detections]
    meanra, ra_error = calculate_stats_coordinates(ra_list, e_ra_list)

    populated_object = object_dto.alerce_object.copy()
    populated_object["meanra"] = meanra
    populated_object["sigmara"] = ra_error

    return ObjectDTO(
        populated_object,
        object_dto.detections,
        object_dto.non_detections
    )
