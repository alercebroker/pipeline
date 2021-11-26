[![codecov](https://codecov.io/gh/alercebroker/survey-parser-plugins/branch/main/graph/badge.svg?token=SEvXaWgJz8)](https://codecov.io/gh/alercebroker/survey-parser-plugins)


# Survey Parser Plugins

Tool to unify content of messages of different surveys for the multi-stream processing.

The generic schema is:

```python
@dataclass
class GenericAlert:
    oid: str # object identifier by telescope. e.g: ZTF20aaelulu
    tid: str # telescope identifier. e.g: ATLAS-01b
    candid: int # candidate identifier by telescope: e.g: 100219327768932647823 
    pid: int # processing identifier for image
    rfid: int # processing identifier for reference image to facilitate archive retrieval
    mjd: float
    fid: int
    ra: float
    e_ra: float # error in right ascension
    dec: float
    e_dec: float # error in declination
    mag: float
    e_mag: float
    isdiffpos: str
    rb: float
    rbversion: str
    extra_fields: dict = field(default_factory=dict)
    stamps: dict = field(default_factory=dict)
```

Where `extra_fields` is a dictionary that has the complement of data that is not generic, that is, the rest of the data that comes from the alert.
The `stamps` is a dictionary with respective cutouts.

The parsers receive a list of messages (messages = dictionaries in python) and return a list of dictionaries with the selected key-value.

## Developer set up

If you want does some modifications on source code, install package as develop in your system.

```
pip install -e .
```

## Usage

For use this package only instance a ParseSelector and register some survey's parser developed by ALeRCE team. The example of below, show you how to use a basic parser that use ATLASParser.

```python
from survey_parser_plugins.core import ParserSelector
from survey_parser_plugins.parsers import ATLASParser

my_parser = ParserSelector()

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

my_parser = ALeRCEParser()

.
.
.

messages_parsed = my_parser.parse(multi_stream_alerts)
```
