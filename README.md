[![codecov](https://codecov.io/gh/alercebroker/survey-parser-plugins/branch/main/graph/badge.svg?token=SEvXaWgJz8)](https://codecov.io/gh/alercebroker/survey-parser-plugins)


# Survey Parser Plugins

Tool to unify content of messages of different surveys for the multi-stream processing.

The basic generic schema is (in progress):

```python
@dataclass
class GenericAlert:
    survey_id: str
    candid: int
    mjd: float
    fid: int
    ra: float
    dec: float
    rb: float
    mag: float
    sigmag: float
    aimage: float
    bimage: float
    xpos: float
    ypos: float
    alerce_id: int = None
    extra_fields: dict = field(default_factory=dict)

```

The main idea is to recover the useful fields for the massive data processing. The parsers are responsible for filtering only useful information from sources:
- ATLAS (version 0.0.1. Since 2021/10/12)
- ZTF (version 0.0.1. Since 2021/10/12)

The parsers receive a list of messages (dictionaries in python) and return a list of dictionaries with the selected key-value.

## Developer set up

If you want does some modifications on source code, install package as develop in your system.

```
pip install -e .
```

## Usage

For use this package only instance a ParseSelector and register some survey's parser developed by ALeRCE team. The example of below, show you how to use a basic parser that use ATLASParse.

```python
from survey_parser_plugins.core import ParserSelector
from survey_parser_plugins.parsers import ATLASParser

my_parser = ParserSelector(alerce_id=True)

""" 
- extra_fields indicates if the parser store more data from alerts in a key called 'extra_fields'
- alerce_id indicates if the parser add alerce_id to parsed message.
"""

my_parser.register_parser(ATLASParser)

.
.
.

messages_parsed = my_parser.parse(list_of_atlas_alerts)
```

Also, we have a custom parsed named `ALeRCEParser` that use all survey's parser that we need.

```python
from survey_parser_plugins.core import ALeRCEParser

my_parser = ALeRCEParser(alerce_id=True)

.
.
.

messages_parsed = my_parser.parse(multi_stream_alerts)
```

An output example of ALeRCEParser can be:

```json
[
  {
    "survey_id": "ZTF18abwhsum",
    "candid": 978463981215010000,
    "mjd": 58732.463981499895,
    "fid": 2,
    "ra": 2.8012592,
    "dec": -12.2713433,
    "rb": 0.47285714745521545,
    "mag": 18.53635597229004,
    "sigmag": 0.12155136466026306,
    "aimage": 0.39500001072883606,
    "bimage": 0.3700000047683716,
    "extra_fields": null,
    "alerce_id": 1001112300121616800
  }, 
  {
    "survey_id": "ZTF20abrnqnv",
    "candid": 1324408541315010000,
    "mjd": 59078.40854169987,
    "fid": 1,
    "ra": 2.3151289,
    "dec": 2.4991734,
    "rb": 0.9428571462631226,
    "mag": 20.411144256591797,
    "sigmag": 0.19233156740665436,
    "aimage": 0.7319999933242798,
    "bimage": 0.6520000100135803,
    "extra_fields": null,
    "alerce_id": 1000915631022957000
  }
]
```
