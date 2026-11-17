# Feature step: `step.py` keeps only the orchestration

## Goal

`feature_step/features/step.py` reads top to bottom as the flow of the step: build
the survey objects, prepare the messages, compute, send, return. The code that does
each of those things in detail (xmatch service, scribe sending, config reading,
message preparation) lives in its own module. Nothing the step consumes, writes or
produces changes.

Today `step.py` has 373 lines. About 230 of them are not orchestration: the xmatch
service code (~130), the same produce-and-flush loop written three times (~75,
counting the xmatch one), config reading (~25), message preparation (~35).

## Non-goals

1. No behaviour change, including the odd parts: messages with `sid == 2` skip the
   xmatch, a match without `catalog` is written as `allwise`, an oid missing from
   the batch gets `sid` 1 in its xmatch command, the flush pattern (below).
2. No change to the scribe, the database, the schemas, the `Survey` contract, the
   charts or the deployed configs.
3. No version bump, no image, no deploy. The running image stays the one of PR #646.
4. The column rename agreed for `feature_version_lut` and `feature` is another PR.
5. `_get_sql_references` (7 lines) and `post_execute` (the `reference` key removal)
   stay in `step.py`: a committed test calls the first, the second is an apf hook of
   four lines.
6. `feature_step/scripts/local_sim` is not refactored. Only the lines that call a
   name this change moves are updated (see criterion 7).

## Design

Four new modules under `feature_step/features/`. Each is plain functions or one
small class, importable and testable without building a `FeatureStep`.

| Module | Holds | Comes from `step.py` |
|---|---|---|
| `scribe_sender.py` | `send_commands(producer, topic, commands)` | the three produce loops |
| `xmatch.py` | `XmatchEnricher` | `_match_for_output`, `get_xmatch_info`, `produce_xmatch_to_scribe`, the attach block of `pre_execute` |
| `settings.py` | `StepSettings`, `read_settings(config)` | the config reads and the `FEATURE_VERSION` validation of `__init__` |
| `messages.py` | `gather_epochs`, `has_enough_detections`, `with_index_column`, `collect_candids` | the second half of `pre_execute`, the loop head of `execute` |

What stays in `step.py`: `__init__` (survey, settings, scribe producer, version id,
LUT, xmatch client), `pre_execute`, `execute`, `produce_to_scribe`, `pre_produce`,
`post_execute`, `_get_sql_references`, `tear_down`.

The version id and the LUT are still resolved in `step.py`, through the names
`get_or_create_version_id` and `get_feature_name_lut` imported there, because the
committed tests and `local_sim/equiv.py` patch them on the step module.

### Approaches considered

1. **Modules of functions next to the step (chosen).** Smallest move: each block is
   cut and pasted behind a function, the step calls it. The tests that build a step
   keep working because the step keeps its attributes.
2. **Push everything into `Survey`.** The xmatch and the scribe sending are the same
   for every survey; putting them on the contract would make each survey carry code
   that is not about the survey.
3. **A pipeline of stage objects** (`Stage.run(batch)`). Reads well, but it changes
   how the step is built and tested for no gain at this size.

## Acceptance criteria

1. `features/scribe_sender.py` has `send_commands(producer, topic, commands)`. For a
   non-empty list it calls `producer.produce(topic=topic, value=json.dumps(command).encode("utf-8"), key=str(command["payload"]["oid"]).encode("utf-8"))`
   once per command, in order, and `producer.flush()` once, after the last one. For
   an empty list it calls neither.
2. `FeatureStep.produce_to_scribe` sends the `update_object` commands and then the
   `upserting_features` commands, with one `send_commands` call per non-empty list:
   two flushes when both lists have commands, one when only one has, none when both
   are empty (the scribe producer is not touched then).
3. `features/xmatch.py` has `XmatchEnricher(client, catalogs, survey_name)` with
   `query(messages)` (one `conesearch_with_metadata` request per catalog, messages
   with `sid == 2` left out, oids as strings), `attach(messages, results)` (sets
   `msg["xmatches"]` to the allwise match with scalar `metadata` entries dropped, or
   `None`) and `commands(results, messages)` (one `xmatch` command per match, same
   keys and defaults as today). The step sends those commands with `send_commands`.
4. `features/settings.py` has `read_settings(config)` returning a `StepSettings`
   with `feature_version`, `min_detections_features`, `use_xmatch`,
   `xmatch_catalogs`, `xmatch_config`, `schema` and `scribe_topic`, with today's
   defaults (`1`, `False`, `["allwise", "gaia"]`, `{}`, `"multisurvey"`, `None`).
   A missing, non-string or blank `FEATURE_VERSION` raises `ValueError` with the
   same three messages as today, before any database call.
5. `features/messages.py` has `gather_epochs(messages, survey)` (shallow copies with
   `"detections"` set from `survey.epochs_from_message`),
   `has_enough_detections(message, minimum)` (counts the epochs whose `forced` is
   falsy), `with_index_column(epochs, id_column)` (adds
   `"<detection id>_<oid>"`) and `collect_candids(messages, id_column)` (oid -> the
   detection ids of all its messages, in order).
6. `FeatureStep` keeps the attributes the committed tests read: `scribe_producer`,
   `xmatch_client` (only when the xmatch is on), `use_xmatch`, `feature_version`,
   `extractor_version`, `feature_name_lut`, and the method `_get_sql_references`.
7. `step.py` has at most 215 lines and none of: `json.dumps`, `.produce(`,
   `conesearch`, `FEATURE_VERSION`. `local_sim/equiv.py` and
   `local_sim/run_step_real_input_nowrite.py` call the moved names at their new
   place and still import.
8. No committed test file changes. Every test that passes on
   `refactor/feature-step-surveys` passes here, the golden tests of
   `test_surveys_golden.py` included.

One note on criterion 1: today the xmatch command is keyed by the oid string of the
match and carries `int(oid)` in its payload. Keying by `str(payload["oid"])` gives
the same bytes for every oid that is a plain integer, which is all of them.

## Done looks like

From the repo root, on the branch:

```bash
~/.pyenv/versions/feature-step/bin/python .claude/scripts/test_delta.py check feature_step
```

exits 0 (the 8 failures of the baseline, no new one), with the new test files of
this change included in the run.

## Branch and PR

`refactor/feature-step-slim`, cut from `refactor/feature-step-surveys`. Draft PR
against that branch, so its diff shows only this change; it moves to `multisurvey`
when #646 is merged.
