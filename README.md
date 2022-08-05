[![codecov](https://codecov.io/gh/alercebroker/consolidated_metrics_step/branch/main/graph/badge.svg?token=XjQI6NzhFB)](https://codecov.io/gh/alercebroker/consolidated_metrics_step)
[![unit_tests](https://github.com/alercebroker/consolidated_metrics_step/actions/workflows/unit_tests.yaml/badge.svg)](https://github.com/alercebroker/consolidated_metrics_step/actions/workflows/unit_tests.yaml)
[![integration_tests](https://github.com/alercebroker/consolidated_metrics_step/actions/workflows/integration_tests.yaml/badge.svg)](https://github.com/alercebroker/consolidated_metrics_step/actions/workflows/integration_tests.yaml)

# Consolidated Metrics Step

A simple step that computes queue, execution and total times.

## Develop set up
1. Install requirements and some libraries to test the step:
```bash
pip install pytest pytest-docker coverage
```
2. To run tests:
```bash
# Unit tests
coverage run --source consolidated_metrics_step/ -m pytest -x -s tests/unit/

# Integration tests: redis and kafka
python -m pytest -x -s tests/integration/
```

3. Install `pre-commit` with
```bash
pip install pre-commit
```

4. Install the git hook scripts. Run pre-commit install to set up the git hook scripts.
```bash
pre-commit install
```

5. When you go to do a commit, the `pre-commit` plugin will verify if satisfies tests, black, imports, etc. The same `pre-commit` will fix your files, and you must do a commit again, but with `pre-commit` changes.

## How to work consolidated metrics?

## Flow of data

## Consolidated metrics considered
