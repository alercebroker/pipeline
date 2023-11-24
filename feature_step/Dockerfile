FROM python:3.10 as python-base
MAINTAINER ALeRCE
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONFAULTHANDLER=1 \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VIRTUALENVS_IN_PROJECT=true \
  POETRY_NO_INTERACTION=1
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y --no-install-recommends git build-essential && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

FROM python-base as builder
WORKDIR /app/
RUN pip install poetry
COPY feature_step/poetry.lock feature_step/pyproject.toml /app/
COPY feature_step/README.md /app/README.md
COPY libs/turbo-fats /libs/turbo-fats
COPY libs/P4J /libs/P4J
COPY libs/mhps /libs/mhps
COPY libs/lc_classifier /libs/lc_classifier
COPY libs/apf /libs/apf
COPY schemas/feature_step /schemas/feature_step
COPY schemas/scribe_step /schemas/scribe_step
RUN poetry install --without=dev,test --no-root
COPY feature_step/scripts /app/scripts
COPY feature_step/settings.py /app/settings.py
COPY feature_step/features /app/features

FROM python:3.10-slim
COPY --from=builder /app /app
COPY --from=builder /libs /libs
COPY --from=builder /schemas /schemas
RUN pip install poetry
WORKDIR /app/
RUN poetry install --only-root
CMD ["poetry", "run", "python", "scripts/run_step.py"]
