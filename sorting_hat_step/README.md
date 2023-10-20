# Sorting hat step
[![codecov](https://codecov.io/gh/alercebroker/sorting_hat_step/branch/main/graph/badge.svg?token=TdPzPGD9Ui)](https://codecov.io/gh/alercebroker/sorting_hat_step)
[![unittest](https://github.com/alercebroker/sorting_hat_step/actions/workflows/unittest.yml/badge.svg)](https://github.com/alercebroker/sorting_hat_step/actions/workflows/unittest.yml)
[![integration_test](https://github.com/alercebroker/sorting_hat_step/actions/workflows/integration.yml/badge.svg)](https://github.com/alercebroker/sorting_hat_step/actions/workflows/integration.yml)

![sorting hat]( https://media.giphy.com/media/JDAVoX2QSjtWU/giphy.gif)

The step of the sorting hat is a step that names the alerts of astronomical survey. The flow of the step is showing in the following image:

TODO: Update figure

1. Find object id in the database: The first query to the database is get the known `oid` by survey. If exists this `oid` in database, the step retrieve the `aid` and assign it to the alert.
2. Cone-search to the database: If the first query hasn't response, the step ask to historical database for nearest objects. If exists the nearest object with a radius of `1.5` arcsec If exists the nearest object with a radius of `1.5` arcsec, the step assign this `aid` to the alert.
3. If there is no `oid` or a nearby object in the database, a new` aid` is created for the alert.


# Development guide

If you make any changes to this repository, run these commands to test your changes (please install `coverage`, `pytest` and `pytest-docker` with pip):

1. Unit tests: Test functionalities with mock of services (kafka, mongo and zookeeper).
```bash
coverage run --source sorting_hat_step -m pytest -x tests/unittest/
```

You can then call `coverage report` to check the coverage.

2. Integration tests: Run the step in an environment with kafka, mongo and zookeeper. This test is useful for developing without setting up a complex environment.

```bash
python -m pytest -x -s tests/integration/
```

## Using Poetry to manage dependencies

Poetry is configured to manage all dependencies in three groups: main, dev and test. 

### Set-up poetry:
- Install poetry: `pip install poetry`
- If you want to set create `.venv` environment in the project folder: `poetry config virtualenvs.in-project true`
- Create environment with all dependencies (main, dev and test): `poetry install`
- To install only main dependencies: `poetry install --only main`
- Show tree of dependencies: `poetry show --tree`
- Add a new dependency 
  - `poetry add PACKAGE`
  - `poetry add -G dev PACKAGE`
  - `poetry add -G test PACKAGE`

### Run tests
- Run all tests : `poetry run pytest`
- Run only unit test: `poetry run pytest tests/unittest`
- Run only integration tests: `poetry run pytest tests/integration`

### Run step
- Run step: `poetry run step`
