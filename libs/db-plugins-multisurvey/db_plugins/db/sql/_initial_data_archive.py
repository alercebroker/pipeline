from typing import Any, TypedDict


class InitData(TypedDict):
    index_elements: list[str]
    data: list[dict[str, Any]]


INITIAL_DATA = {
    "classifier_version": {
        "index_elements": ["id"],
        "data": [
            {
                "id": 1,
                "classifier_name": "stamp_classifier_rubin",
                "classifier_version": "2.0.1",
                "features_version": "0",
            },
            {
                "id": 2,
                "classifier_name": "stamp_classifier_2025_beta",
                "classifier_version": "2.1.1",
                "features_version": "0",
            },
        ],
    },
    "sid_lut": {
        "index_elements": ["sid"],
        "data": [
            {"sid": 0, "tid": 0, "survey_name": "ZTF"},
            {"sid": 1, "tid": 1, "survey_name": "LSST DIA Object"},
            {"sid": 2, "tid": 1, "survey_name": "LSST SS Object"},
        ],
    },
    "taxonomy": {
        "index_elements": ["id"],
        "data": [
            {"id": 0, "class_name": "SN", "order": 0, "classifier_version_id": 1},
            {"id": 1, "class_name": "AGN", "order": 1, "classifier_version_id": 1},
            {"id": 2, "class_name": "VS", "order": 2, "classifier_version_id": 1},
            {"id": 3, "class_name": "asteroid", "order": 3, "classifier_version_id": 1},
            {"id": 4, "class_name": "bogus", "order": 4, "classifier_version_id": 1},
            {"id": 5, "class_name": "SN", "order": 0, "classifier_version_id": 2},
            {"id": 6, "class_name": "AGN", "order": 1, "classifier_version_id": 2},
            {"id": 7, "class_name": "VS", "order": 2, "classifier_version_id": 2},
            {"id": 8, "class_name": "asteroid", "order": 3, "classifier_version_id": 2},
            {"id": 9, "class_name": "bogus", "order": 4, "classifier_version_id": 2},
            {"id": 10, "class_name": "satellite", "order": 5, "classifier_version_id": 2},
        ],
    },
}