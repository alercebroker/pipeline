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
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY metadata_step/poetry.lock metadata_step/pyproject.toml /app/
COPY libs/apf libs/apf
COPY libs/db-plugins libs/db-plugins
COPY schemas/sorting_hat_step /schemas/sorting_hat_step
COPY schemas/metadata_step /schemas/metadata_step
COPY schemas/scribe_step /schemas/scribe_step

WORKDIR /app
RUN poetry install --no-root --without=dev --without=test

FROM python:3.10-slim as production
RUN pip install poetry

COPY --from=builder /app /app
COPY --from=builder /libs /libs
COPY --from=builder /schemas /schemas

COPY metadata_step/metadata_step /app/metadata_step
COPY metadata_step/scripts /app/scripts
COPY metadata_step/README.md \
    metadata_step/settings.py \
    metadata_step/scribe_schema.avsc \
    metadata_step/credentials.py \
    /app/

WORKDIR /app
RUN poetry install --only-root

CMD ["poetry", "run", "python", "scripts/run_step.py"]
