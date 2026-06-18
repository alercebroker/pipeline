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

# Optional image-build overlay: override confluent-kafka in the in-project venv.
# No-op when the build-arg is empty, so default (AWS) builds are unchanged. Needed
# for the quimal SCRAM-SHA-512 broker: the lock pins confluent-kafka ~2.0.2, whose
# librdkafka 2.0.2 has a SCRAM bug surfacing as a bogus "invalid credentials"
# (same blocker as the watchlist-step -cfk2.14.2 overlay). This is an image-layer
# bump, NOT a poetry re-lock (re-locking conflicts with apf multisurvey; see
# aws_cost_analysis/docs/metadata-step-revival-plan.md §7).
ARG CONFLUENT_KAFKA_VERSION=
RUN if [ -n "$CONFLUENT_KAFKA_VERSION" ]; then \
      /app/.venv/bin/pip install --no-cache-dir --upgrade "confluent-kafka==$CONFLUENT_KAFKA_VERSION" && \
      /app/.venv/bin/python -c "import confluent_kafka as k; print('cfk overlay:', k.__version__, 'librdkafka', k.libversion())"; \
    fi

CMD ["poetry", "run", "python", "scripts/run_step.py"]
