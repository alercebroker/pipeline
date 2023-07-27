FROM python:3.9-slim as python-base
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
RUN pip install poetry

COPY alert_archiving_step/poetry.lock alert_archiving_step/pyproject.toml /app/
COPY libs/apf /libs/apf

WORKDIR /app
RUN poetry install --no-root --without=dev --without=test

FROM python:3.9-slim as production
RUN pip install poetry

COPY --from=builder /app /app
COPY --from=builder /libs /libs

COPY alert_archiving_step/alert_archiving_step /app/alert_archiving_step
COPY alert_archiving_step/scripts /app/scripts
COPY alert_archiving_step/README.md /app

WORKDIR /app
RUN poetry install --no-interaction --only-root

CMD ["poetry", "run", "run-step"]
