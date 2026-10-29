# Stamp step Dockerfile: `model` build arg default reaches the builder stage

Date: 2026-10-29. PR #633 checklist item. Two-line change, one test.

## Goal
`docker build` of `rubin_stamp_classifier_step/Dockerfile` without `--build-arg model=...` installs the rubin dependency group, as the default `ARG model=rubin` intends. Today that default is scoped to the first stage and the builder stage re-declares `ARG model` empty, so `poetry install --with=` runs with no group.

## Non-goals
Any other Dockerfile change; CI workflow changes (CI already passes the arg).

## Acceptance criteria
1. `ARG model=rubin` is declared before the first `FROM` (global scope) and the builder stage keeps a bare `ARG model` to pick it up. `poetry install --no-root --without=dev --with=${model}` unchanged.
2. New unit test `rubin_stamp_classifier_step/tests/unit/test_dockerfile_model_arg.py`: the first non-comment, non-empty instruction of the Dockerfile is `ARG model=rubin`; a bare `ARG model` appears in the stage that runs `poetry install`; the install line references `--with=${model}`.

## Done looks like
`test_delta.py check rubin_stamp_classifier_step` exit 0 (baseline 0). Image build not exercised here (docker not required for the unit suite).
