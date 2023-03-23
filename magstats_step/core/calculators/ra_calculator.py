from ..utils.object_dto import ObjectDTO
from .coordinate_calculator import calculate_stats_coordinates


def calculate_ra(object_dto: ObjectDTO) -> ObjectDTO:
    ra_series, e_ra_series = object_dto.detections["ra"], object_dto.detections["e_ra"]
    meanra, ra_error = calculate_stats_coordinates(ra_series, e_ra_series)

    populated_object = object_dto.alerce_object.copy()
    populated_object["meanra"] = meanra
    populated_object["sigmara"] = ra_error

    return ObjectDTO(
        populated_object,
        object_dto.detections,
        object_dto.non_detections,
        object_dto.extra_fields
    )
