# LSST Schema Parser

Converters that upgrade an LSST alert from one Rubin alert schema version to the
next. Each converter takes an alert as a Python `dict` in the older schema and
returns a new `dict` in the newer one.

| Module (`schema_parser/parsers/`) | Class                 | Converts      |
| --------------------------------- | --------------------- | ------------- |
| `v9_to_v10.py`                    | `V9ToV10Parser`       | v9 → v10      |
| `v10_to_v11.py`                   | `V10ToV11Parser`      | v10 → v11     |
| `v11_0_to_v11_1.py`               | `V11_0ToV11_1Parser`  | v11.0 → v11.1 |
| `v11_1_to_v11_2.py`               | `V11_1ToV11_2Parser`  | v11.1 → v11.2 |

Each class's docstring describes what changed between its two versions.

## Usage

Converters do one version step. To go further, apply them in order:

```python
from schema_parser.parsers.v9_to_v10 import V9ToV10Parser
from schema_parser.parsers.v10_to_v11 import V10ToV11Parser
from schema_parser.parsers.v11_0_to_v11_1 import V11_0ToV11_1Parser
from schema_parser.parsers.v11_1_to_v11_2 import V11_1ToV11_2Parser

alert = ...  # a v9 alert as a dict
for parser in (V9ToV10Parser(), V10ToV11Parser(), V11_0ToV11_1Parser(), V11_1ToV11_2Parser()):
    alert = parser.parse(alert)
```

## Step wrapper

`schema_parser/step.py` wraps a converter in an APF step, and `scripts/run_step.py`
starts it. It is not usable as is: `run_step.py` reads a `config.yaml` that does
not exist in this library. No Dockerfile, CI job or chart builds or deploys it.

## Tests

The unit tests validate converted alerts against the Avro schemas in the
repository's top-level `schemas/surveys/` folder, so run them from a full checkout.
Use Python 3.10: the `fastavro` version pinned by `apf` has no Python 3.11 build,
so `poetry install` fails on 3.11.

```bash
poetry install
poetry run pytest tests/unit
```
