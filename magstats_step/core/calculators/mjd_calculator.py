from ..utils.object_dto import ObjectDTO

def calculate_mjd(object_dto: ObjectDTO) -> ObjectDTO:
    detections = object_dto.detections
    if detections.empty:
        return object_dto

    populated_object = object_dto.alerce_object.copy()
    populated_object["firstmjd"] = detections["mjd"].min()
    populated_object["lastmjd"] = detections["mjd"].max()

    return ObjectDTO(
        populated_object,
        detections,
        object_dto.non_detections,
        object_dto.extra_fields
    )
