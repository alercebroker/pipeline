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

COPY correction_multistream_ztf_step/pyproject.toml /app/
COPY correction_multistream_ztf_step/poetry.lock /app/
COPY libs/apf libs/apf
COPY libs/db-plugins-multisurvey libs/db-plugins-multisurvey
COPY libs/idmapper libs/idmapper
COPY libs/survey_parser_plugins libs/survey_parser_plugins

COPY schemas/ingestion_step schemas/ingestion_step
COPY schemas/correction_ms_ztf schemas/correction_ms_ztf
COPY schemas/scribe_step schemas/scribe_step

WORKDIR /app
RUN poetry install --no-root --without=dev --without=test;

FROM python:3.10-slim AS production
RUN pip install poetry;

COPY --from=builder /app /app
COPY --from=builder /libs /libs
COPY --from=builder /schemas /schemas

COPY correction_multistream_ztf_step/README.md \
  /app/

COPY correction_multistream_ztf_step/README.md /app/
COPY correction_multistream_ztf_step/correction_multistream_ztf_step /app/correction_multistream_ztf_step
COPY correction_multistream_ztf_step/core /app/core
COPY correction_multistream_ztf_step/settings.py /app/settings.py

COPY correction_multistream_ztf_step/scripts /app/scripts
COPY correction_multistream_ztf_step/credentials.py /app/
WORKDIR /app
RUN poetry install --only-root;

CMD ["poetry", "run", "python", "scripts/run_step.py"]
