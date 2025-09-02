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

RUN pip install poetry;

COPY libs/alerts_store/poetry.lock libs/alerts_store/pyproject.toml /libs/alerts_store/

COPY libs/apf /libs/apf

COPY schemas/ztf /schemas/ztf
COPY schemas/surveys /schemas/surveys

WORKDIR /libs/alerts_store
RUN poetry install --no-root --without=dev;

FROM python:3.10-slim AS production
RUN pip install poetry;

COPY --from=builder /libs /libs
COPY --from=builder /schemas /schemas

COPY libs/alerts_store/alerts_store /libs/alerts_store/alerts_store
COPY libs/alerts_store/scripts /libs/alerts_store/scripts
COPY libs/alerts_store/README.md /libs/alerts_store/README.md

WORKDIR /libs/alerts_store
RUN poetry install --only-root;

CMD ["poetry", "run", "step"]
