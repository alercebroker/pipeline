from ..utils.object_dto import ObjectDTO


def calculate_ndet(object_dto: ObjectDTO) -> ObjectDTO:
    detections = object_dto.detections
    populated_object = object_dto.alerce_object.copy()
    populated_object["ndet"] = len(detections.index)

    return ObjectDTO(
        populated_object,
        object_dto.detections,
        object_dto.non_detections,
        object_dto.extra_fields
    )