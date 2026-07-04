# MongoDB in this repo is a retired backend

**TL;DR.** MongoDB was ALeRCE's *original* multi-survey storage attempt. It was
abandoned in favour of PostgreSQL (the schema the current multisurvey pipeline
uses). **No MongoDB instance runs anywhere today** — not for the legacy ZTF
pipeline, not for the multisurvey one. The Mongo code that is still scattered
through this repo is **dead weight left in place on purpose**: the legacy
pipeline is being retired, so removing it isn't worth the churn. This document
exists so you don't mistake the fossils for a live system.

Last reviewed: 03/07/2026.

---

## Why you'll see "mongo" everywhere

A case-insensitive search for `mongo` hits ~30 source/config/doc files (plus a
lot of `pymongo` wheel hashes in `poetry.lock`s). The meaningful ones:

| Area | What it is | Status |
|---|---|---|
| `libs/db-plugins/db_plugins/db/mongo/` | Full Mongo ORM: `_connection.py`, `models.py`, `orm.py`, `initialization.py`, `helpers/` | Unused at runtime |
| `scribe/` (the package is literally `mongo_scribe`, class `MongoScribe`) | The scribe supports **both** a Mongo and a SQL backend, chosen at runtime | Only the **SQL** path is deployed |
| `scribe/mongo_scribe/mongo/` | The Mongo half of the scribe (command decode + executor) | Unused at runtime |
| `lightcurve-step` | Parallel `database_mongo.py`/`parser_mongo.py` **and** `database_sql.py`/`parser_sql.py`, toggled by `USE_MONGO`/`USE_SQL` | Mongo path off |
| `sorting_hat_step` | `MongoConnection` + `MONGO_CONFIG` (Mongo once held the cross-survey `aid` identity) | Postgres (`ENGINE: "postgres"`) |
| `correction_multisurvey_step` | Still declares `pymongo` + references it in `credentials.py` | Vestigial |
| `pymongo` / `mongomock` deps | Pinned in `db-plugins`, `test_utils`, `sorting_hat`, `lightcurve`, `correction_multisurvey` | Installed, mostly for tests |
| `tests/**/docker-compose.*`, `.github/workflows/*` | Integration tests spin up a **throwaway** Mongo container | The only place a real Mongo actually starts |

The naming is a fossil: `mongo_scribe`/`MongoScribe` predate the SQL backend and
were never renamed.

## How Mongo is gated off

Nothing needs Mongo removed because every deployment simply doesn't select it:

- **scribe** — `DB_TYPE = os.getenv("DB_ENGINE", "mongo")` (`scribe/settings.py`)
  *defaults* to mongo, but the deployed config sets `DB_ENGINE=sql`, and
  `scribe/scripts/run_step.py` then builds `MongoScribe(..., db="sql")`, wiring
  the `sql/` command factory + `SQLCommandExecutor`.
- **lightcurve / sorting_hat** — the Helm charts set `MONGO_SECRET_NAME: ""`, so
  no Mongo credentials are provided and the Mongo code paths are inert.

## The consequence that actually bites: dual-dialect scribe commands

This is the reason this doc exists. ALeRCE's scribe layer is a **CQRS-style
writer** with two command *dialects*, one per historical backend:

- **Mongo dialect** — generic document ops keyed on `type`
  (`insert` / `update` / `update_probabilities` / `update_features`), any
  `collection`. Decoded by `scribe/mongo_scribe/mongo/command/decode.py`.
- **SQL dialect** — typed, relational ops matched on `(type, collection)`
  pairs. Decoded by `scribe/mongo_scribe/sql/command/decode.py`.

Some producers still emit **both dialects for the same write**, so the step
could feed either backend. The clearest case is **`magstats_step`**
([`magstats_step/step.py`](magstats_step/magstats_step/step.py)): its
`post_execute` calls `produce_scribe()` **and** `produce_scribe_ztf()`, sending
two commands per object built from the *same* stats dict:

```
produce_scribe      → {"type":"update",  "collection":"object"}   # Mongo dialect
produce_scribe_ztf  → {"type":"upsert",  "collection":"magstats"} # SQL dialect
```

The SQL scribe (`scribe-psql`, the only one deployed on quimal) consumes the
`upsert`/`magstats` command — `UpdateObjectStatsCommand` writes the object
aggregates to the `object` table **and** upserts the per-band `magstats` rows.
The `update`/`object` command is the **Mongo dialect**: it has no SQL handler,
and with no Mongo backend to consume it, it is an **orphan** — produced, then
dropped by the only scribe reading the topic. **No data is lost**: the identical
statistics land via the SQL command.

Historically this orphan was logged (at DEBUG) as
`Unrecognized command type update in table object.` and counted in the
`Found N invalid messages` INFO summary. That counter therefore **conflated** two
very different things: these frequent, benign Mongo-dialect orphans, and genuine
dropped commands (e.g. the `parent_candid="nan"` detection drop). If you're using
`Found N invalid` to size real data loss, subtract the Mongo-dialect skips first.

## What we did about it (and what we deliberately did not)

- **Did not** remove the Mongo deadweight — the legacy pipeline is retiring, so
  it isn't worth the churn.
- **Did** teach the SQL decoder to recognise the Mongo-dialect object update
  explicitly and skip it *quietly*:
  - `sql/command/decode.py` raises the dedicated
    `MongoDialectCommandException` (subclass of `ValueError`) for a
    `{"type":"update","collection":"object"}` command with no `xmatch`.
  - `scribe/mongo_scribe/step.py` catches that **before** the generic
    `except ValueError`, counts it separately, and reports it as
    `Skipped N legacy Mongo-dialect commands` instead of WARN-logging each with
    its full payload.
  - Why this matters now: a companion fix raised the previously-silent
    invalid-command drop from DEBUG to **WARN + payload** (so real losses like
    the `nan` detection stop vanishing silently). Without this skip, the
    per-object magstats orphan would **flood** that WARN channel and bury the
    signal it was added to surface.

If a future SQL-relevant object update is ever needed, give it a **distinct
type** (e.g. the existing `update_object_from_stats`) rather than reusing the
generic Mongo `update`/`object` shape.

## See also

- `scribe/README.md` — scribe command format (Mongo-centric; the SQL dialect is
  the `sql/command/` decoder).
- Detection-loss / `parent_candid="nan"` investigation (aws_cost_analysis repo,
  `docs/detection-loss-debug-plan.md`) — where the `Found N invalid` counter
  reconciliation was traced.
