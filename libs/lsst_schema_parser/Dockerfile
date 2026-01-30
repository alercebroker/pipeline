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
RUN apt-get update && \
    apt-get install -y --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry;

COPY ingestion_step/poetry.lock ingestion_step/pyproject.toml /app/

COPY libs/apf libs/apf
COPY libs/db-plugins-multisurvey libs/db-plugins-multisurvey
COPY libs/idmapper libs/idmapper
COPY libs/test_utils libs/test_utils
COPY libs/survey_parser_plugins libs/survey_parser_plugins

COPY schemas/ztf schemas/ztf
COPY schemas/ingestion_step schemas/ingestion_step
COPY schemas/surveys schemas/surveys

WORKDIR /app
RUN poetry install --no-root --without=dev --without=test;

FROM python:3.10-slim AS production
RUN pip install poetry;

COPY --from=builder /app /app
COPY --from=builder /libs /libs
COPY --from=builder /schemas /schemas

COPY ingestion_step/generator /app/generator
COPY ingestion_step/ingestion_step /app/ingestion_step
COPY ingestion_step/scripts /app/scripts
COPY ingestion_step/README.md /app/

WORKDIR /app
RUN poetry install --only-root;

CMD ["poetry", "run", "step"]
