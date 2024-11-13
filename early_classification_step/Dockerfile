FROM python:3.7-slim as python-base
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
  apt-get install -y --no-install-recommends git libpq-dev gcc && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

COPY early_classification_step/poetry.lock early_classification_step/pyproject.toml /app/
COPY early_classification_step/model /app/model
WORKDIR /app

FROM python:3.7-slim as production

COPY --from=builder /app /app

COPY schemas/early_classification_step /schemas/early_classification_step
COPY early_classification_step/earlyclassifier /app/earlyclassifier
COPY early_classification_step/scripts /app/scripts
COPY early_classification_step/model /app/model
COPY early_classification_step/README.md \
  early_classification_step/settings.py \
  /app/
COPY early_classification_step/requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip install -r requirements.txt

CMD ["python", "scripts/run_step.py"]
