[![codecov](https://codecov.io/gh/alercebroker/survey-parser-plugins/branch/main/graph/badge.svg?token=SEvXaWgJz8)](https://codecov.io/gh/alercebroker/survey-parser-plugins)


# Survey Parser Plugins

Tool to unify content of messages of different surveys for the multi-stream processing.

The basic generic schema is (in progress):

```yml
Entities:
  Object:
    - alerce_id: str
    - survey_id: str
    - lastmjd: float
    - firstmjd: float
    - meanra: float
    - meandec: float
    - sigmara: float
    - sigmadec: float
  Detection:
    - alerce_id: str
    - survey_id: str
    - candid : str
    - mjd : float
    - fid : int
    - ra : float
    - dec : float
    - rb: float
    - mag: float
    - sigmag: float
    - extra_fields: JSON
  Non-Detection:
    - alerce_id: str
    - survey_id: str
    - mjd: float
    - diffmaglim: float
    - fid: int
    - extra_fields: JSON
  Classification:
    - alerce_id: str
    - survey_id: str
    - classifier_name: str
    - classifier_version: str
    - class_name: str
    - probabilty: float
    - ranking: JSON
    - probabilities: JSON
    - last_updated: date
    - created_on: date
  Features:
    - alerce_id: str
    - survey_id: str
    - features: JSON
    - created_on: date
    - last_updated: date
  Xmatch:
    - alerce_id: str
    - survey_id: str
    - catalog_name: str
    - catalog_id: str
```

The main idea is to recover the useful fields for the massive data processing. The parsers are responsible for filtering only useful information from sources:
- ATLAS (version 0.0.1. Since 2021/10/12)
- ZTF (version 0.0.1. Since 2021/10/12)

The parsers receive a list of messages (dictionaries in python) and return a list of dictionaries with the selected key-value.

