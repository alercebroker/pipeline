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
  apt-get install -y --no-install-recommends git && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*;

RUN pip install poetry;

COPY scribe_multisurvey/poetry.lock scribe_multisurvey/pyproject.toml /app/
COPY libs/apf libs/apf
COPY libs/db-plugins-multisurvey libs/db-plugins-multisurvey
COPY schemas/scribe_step /schemas/scribe_step

WORKDIR /app
RUN poetry install --no-root --without=dev --without=test;

FROM python:3.10-slim AS production
RUN pip install poetry;

COPY --from=builder /app /app
COPY --from=builder /libs /libs
COPY --from=builder /schemas /schemas

COPY scribe_multisurvey/sql_scribe /app/sql_scribe
COPY scribe_multisurvey/scripts /app/scripts
COPY scribe_multisurvey/README.md \
  scribe_multisurvey/credentials.py \
  /app/

WORKDIR /app
RUN poetry install --only-root;

CMD ["poetry", "run", "python", "scripts/run_step.py"]
