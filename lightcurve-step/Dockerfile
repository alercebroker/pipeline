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

COPY lightcurve-step/poetry.lock lightcurve-step/pyproject.toml /app/
COPY libs/apf libs/apf
COPY libs/db-plugins libs/db-plugins
COPY schemas/prv_candidate_step /schemas/prv_candidate_step
COPY schemas/lightcurve_step /schemas/lightcurve_step
COPY schemas/scribe_step /schemas/scribe_step

WORKDIR /app
RUN poetry install --no-root --without=dev --without=test


FROM python:3.10-slim as production
RUN pip install poetry

COPY --from=builder /app /app
COPY --from=builder /libs /libs
COPY --from=builder /schemas /schemas

COPY lightcurve-step/lightcurve_step /app/lightcurve_step
COPY lightcurve-step/scripts /app/scripts
COPY lightcurve-step/README.md \
  lightcurve-step/settings.py \
  lightcurve-step/credentials.py \
  /app/

WORKDIR /app
RUN poetry install --only-root

CMD ["poetry", "run", "python", "scripts/run_step.py"]
