FROM python:3.10-slim AS python-base
LABEL org.opencontainers.image.authors="ALeRCE"

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONFAULTHANDLER=1 \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VIRTUALENVS_IN_PROJECT=true \
  POETRY_NO_INTERACTION=1

FROM python-base AS builder
RUN pip install poetry

COPY hips_dummy_step/poetry.lock hips_dummy_step/pyproject.toml /app/
COPY libs/apf /libs/apf

WORKDIR /app
RUN poetry install --no-root --without=dev --without=test

FROM python-base AS production
RUN pip install poetry

COPY --from=builder /app /app
COPY --from=builder /libs /libs
# LSST alerts are read with a schema; ZTF alerts carry their own.
COPY schemas/surveys /schemas/surveys

COPY hips_dummy_step/hips_dummy_step /app/hips_dummy_step
COPY hips_dummy_step/scripts /app/scripts
COPY hips_dummy_step/README.md /app/

WORKDIR /app
RUN poetry install --only-root

CMD ["poetry", "run", "python", "scripts/run_step.py"]
