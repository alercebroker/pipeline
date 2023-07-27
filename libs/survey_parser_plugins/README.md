[![codecov](https://codecov.io/gh/alercebroker/survey-parser-plugins/branch/main/graph/badge.svg?token=SEvXaWgJz8)](https://codecov.io/gh/alercebroker/survey-parser-plugins)


# Survey Parser Plugins

Tool to unify content of messages of different surveys for the multi-stream processing.

The generic schema is:

```python
@dataclass
class GenericAlert:
    oid: str  # name of object (from survey)
    tid: str  # telescope identifier
    sid: str  # survey identifier
    pid: int  # processing identifier for image
    candid: str  # candidate identifier (from survey)
    mjd: float  # modified Julian date
    fid: str  # filter identifier
    ra: float  # right ascension
    dec: float  # declination
    mag: float  # difference magnitude
    e_mag: float  # difference magnitude uncertainty
    isdiffpos: int  # sign of the flux difference
    e_ra: float = None  # right ascension uncertainty
    e_dec: float = None  # declination uncertainty
    extra_fields: dict = field(default_factory=dict)
    stamps: dict = field(default_factory=dict)
```

Where `extra_fields` is a dictionary that has the complement of data that is not generic, that is, the rest of the data that comes from the alert.

The `stamps` is a dictionary with `science`, `template` and `difference`. One or more can be `None` if not
provided by the source.

The parsers receive a list of messages (dictionaries in python) and return a list of generic alerts.

## Developer set up

If you want does some modifications on source code, install package as develop in your system.

```
pip install -e .[dev]
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
