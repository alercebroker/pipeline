FROM python:3.10-slim as python-base
LABEL org.opencontainers.image.authors="ALeRCE"

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONFAULTHANDLER=1 \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VIRTUALENVS_IN_PROJECT=true \
  POETRY_NO_INTERACTION=1

FROM python-base as builder

RUN pip install poetry;
COPY ingestion_step/poetry.lock ingestion_step/pyproject.toml /app/
COPY libs/apf libs/apf
COPY libs/db-plugins libs/db-plugins
COPY libs/survey_parser_plugins libs/survey_parser_plugins

WORKDIR /app
RUN poetry install --no-root --without=dev --without=test;

FROM python:3.10-slim as production
RUN pip install poetry;

COPY --from=builder /app /app
COPY --from=builder /libs /libs
COPY --from=builder /schemas /schemas

COPY ingestion_step/ingestion_step /app/ingestion_step
COPY ingestion_step/scripts /app/scripts
COPY ingestion_step/credentials.py \
  ingestion_step/settings.py \
  ingestion_step/README.md \
  /app/

WORKDIR /app
RUN poetry install --only-root;

CMD ["poetry", "run", "python", "scripts/run_step.py"]
