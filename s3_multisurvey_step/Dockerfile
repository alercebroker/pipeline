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

RUN pip install poetry

COPY libs/apf libs/apf
COPY ./s3_multisurvey_step/poetry.lock ./s3_multisurvey_step/pyproject.toml /app/
COPY schemas/surveys /schemas/surveys
COPY schemas/ztf /schemas/ztf
COPY schemas/s3_ms_step /schemas/s3_ms_step

WORKDIR /app
RUN poetry install --no-root --without=dev --without=test

FROM python:3.10-slim AS production
RUN pip install poetry

COPY --from=builder /app /app
COPY --from=builder /libs /libs
COPY --from=builder /schemas /schemas

COPY s3_multisurvey_step/scripts /app/scripts
COPY s3_multisurvey_step/s3_step /app/s3_step
COPY s3_multisurvey_step/README.md /app/


WORKDIR /app
RUN poetry install --only-root

CMD ["poetry", "run", "python", "scripts/run_step.py"]
