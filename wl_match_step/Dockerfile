FROM python:3.11-slim as python-base
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
  apt-get install -y --no-install-recommends gcc libpq-dev python3-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY wl_match_step/poetry.lock wl_match_step/pyproject.toml /app/

WORKDIR /app
RUN poetry install --no-root --without=dev --without=test


FROM python:3.11-slim as production
RUN pip install poetry

COPY --from=builder /app /app
# COPY --from=builder /libs /libs

COPY wl_match_step/wl_match_step /app/wl_match_step
COPY wl_match_step/scripts /app/scripts
COPY wl_match_step/README.md \
  wl_match_step/settings.py \
  /app/

WORKDIR /app
RUN poetry install --only-root

CMD ["poetry", "run", "step"]
