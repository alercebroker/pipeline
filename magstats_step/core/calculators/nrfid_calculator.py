import numpy as np
from ..utils.object_dto import ObjectDTO

def calculate_nrfid(object_dto: ObjectDTO):
    detections_df = object_dto.extra_fields
    nrfids = detections_df["rfid"].unique().astype(float)
    nrfids = nrfids[~np.isnan(nrfids)]
    populated_object = object_dto.alerce_object.copy()

    populated_object["magstats"].append({
        "name": "nrfid",
        "value": len(nrfids)
    })

    return ObjectDTO(
        populated_object,
        object_dto.detections,
        object_dto.non_detections,
        object_dto.extra_fields
    )
