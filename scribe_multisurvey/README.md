# ALeRCE Scribe Multisurvey

This step consumes Kafka messages produced by upstream pipeline steps (correction, magstats, features, xmatch) and writes processed data into a PostgreSQL database via SQLAlchemy.

## Overview

`SqlScribe` extends `GenericStep` and on each batch:
1. **`pre_execute`** — deduplicates magstat messages, keeping the most up-to-date record per object (by `n_det`/`n_fphot`/`ndubious` counts for ZTF, by `lastmjd` for LSST).
2. **`execute`** — decodes each message's `payload` field into a typed `Command` object, then bulk-executes all valid commands against the database. Invalid messages are logged and skipped.

## Message Format

Each Kafka message must have a `payload` field containing a stringified JSON with the following structure:

```json
{
    "survey": "ztf",
    "step": "correction",
    "payload": { ... }
}
```

The `survey` and `step` fields together determine which `Command` subclass handles the message.

## Supported Commands

| `survey` | `step`       | Command class           | Tables written                                                                                                          |
|----------|--------------|-------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `ztf`    | `correction` | `ZTFCorrectionCommand`  | `detection`, `ztf_detection`, `forced_photometry`, `ztf_forced_photometry`, `ztf_ps1`, `ztf_ss`, `ztf_gaia`, `ztf_dataquality`, `ztf_reference`, `ztf_object` |
| `ztf`    | `magstat`    | `ZTFMagstatCommand`     | `object` (update), `ztf_object` (update), `magstat` (upsert)                                                            |
| `lsst`   | `magstat`    | `LSSTMagstatCommand`    | `object` (update)                                                                                                       |
| `lsst`   | `features`   | `LSSTFeatureCommand`    | `feature` (upsert)                                                                                                      |
| `ztf`/`lsst` | `xmatch` | `XmatchCommand`         | `xmatch` (upsert); sid=2 (SS Object) is skipped                                                                        |

## Configuration

The step requires a `PSQL_CONFIG` key in its config dict:

```python
config = {
    "PSQL_CONFIG": {
        "USER": "...",
        "PASSWORD": "...",
        "HOST": "...",
        "PORT": 5432,
        "DB_NAME": "...",
        "SCHEMA": "public",      # optional
        "POOLCLASS": "NullPool", # recommended for pgbouncer
    },
    "CONSUMER_CONFIG": { ... },
    "PRODUCER_CONFIG": { ... },
}
```

See [settings.py](settings.py) for a full example using environment variables.

## Suggested Producer Schema

For upstream steps sending data to the scribe, use:

```python
SCRIBE_PRODUCER_CONFIG = {
    "TOPIC": os.environ["SCRIBE_TOPIC"],
    "PARAMS": {
        "bootstrap.servers": os.environ["SCRIBE_SERVER"],
    },
    "SCHEMA": {
        "namespace": "db_operation",
        "type": "record",
        "name": "Command",
        "fields": [
            {"name": "payload", "type": "string"},
        ],
    },
}
```

## Internal Structure

```
sql_scribe/
  step.py                    # SqlScribe step (pre_execute, execute)
  sql/
    command/
      decode.py              # decode_message(), command_factory()
      commands.py            # Command base class + all Command subclasses
      parser.py              # Field-level data transformation helpers
      exceptions.py          # Custom exceptions
    db/
      connection.py          # PsqlDatabase wrapper
      executor.py            # SQLCommandExecutor (bulk_execute)
```

## Running

```bash
python scripts/run_step.py
```

