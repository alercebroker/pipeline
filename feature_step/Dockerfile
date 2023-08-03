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
RUN pip install poetry numpy Cython
COPY ./poetry.lock ./pyproject.toml /app/
COPY ./README.md /app/README.md
RUN poetry install --no-interaction --without=dev --without=test --no-root
COPY ./scripts /app/scripts
COPY ./settings.py /app/settings.py
COPY ./schema.py /app/schema.py
COPY ./scribe_schema.avsc /app/scribe_schema.avsc
COPY ./features /app/features

FROM python:3.10-slim
WORKDIR /app/
RUN pip install poetry
COPY --from=builder /app /app
RUN poetry install --no-interaction --only-root
CMD ["poetry", "run", "python", "scripts/run_step.py"]
