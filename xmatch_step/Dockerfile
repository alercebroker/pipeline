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

COPY xmatch_step/poetry.lock xmatch_step/pyproject.toml /app/
COPY libs/apf /libs/apf
COPY schemas/correction_step /schemas/correction_step
COPY schemas/xmatch_step /schemas/xmatch_step
COPY schemas/scribe_step /schemas/scribe_step

WORKDIR /app
RUN poetry install --no-root --without=dev --without=test


FROM python:3.10-slim as production
RUN pip install poetry

COPY --from=builder /app /app
COPY --from=builder /libs /libs
COPY --from=builder /schemas /schemas

COPY xmatch_step/xmatch_step /app/xmatch_step
COPY xmatch_step/scripts /app/scripts
COPY xmatch_step/README.md \
  xmatch_step/settings.py \
  /app/

WORKDIR /app
RUN poetry install --only-root

CMD ["poetry", "run", "step"]
