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
RUN pip install poetry
WORKDIR /app
COPY ./poetry.lock ./pyproject.toml ./
RUN poetry install --no-root --without=dev --without=test


FROM python:3.7-slim as production
RUN pip install poetry
COPY --from=builder /app /app
WORKDIR /app
COPY ./earlyclassifier /app/earlyclassifier
COPY ./model /app/model
COPY ./scripts /app/scripts
COPY ./settings.py /app/settings.py
COPY ./README.md /app/README.md
RUN poetry install --only-root
CMD ["poetry", "run", "python", "scripts/run_step.py"]
