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

COPY magstats_multisurvey_step/poetry.lock magstats_multisurvey_step/pyproject.toml /app/
COPY libs/apf libs/apf
COPY schemas/magstats_ms_step /schemas/magstats_ms_step
COPY schemas/correction_ms_step /schemas/correction_ms_step
COPY schemas/scribe_step /schemas/scribe_step

WORKDIR /app
RUN poetry install --no-root --without=dev --without=test

FROM python:3.10-slim as production
RUN pip install poetry

COPY --from=builder /app /app
COPY --from=builder /libs /libs
COPY --from=builder /schemas /schemas

COPY magstats_multisurvey_step/README.md /app/README.md
COPY magstats_multisurvey_step/scripts /app/scripts
COPY magstats_multisurvey_step/settings.py /app/settings.py
COPY magstats_multisurvey_step/magstats_multisurvey_step /app/magstats_multisurvey_step

WORKDIR /app

RUN poetry install --only-root

CMD ["poetry", "run", "python", "scripts/run_step.py"]
