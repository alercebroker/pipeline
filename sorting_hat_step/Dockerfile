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

RUN pip install poetry
COPY libs/survey_parser_plugins libs/survey_parser_plugins
COPY sorting_hat_step/poetry.lock sorting_hat_step/pyproject.toml /app/
COPY libs/apf libs/apf
COPY libs/db-plugins libs/db-plugins
COPY libs/survey_parser_plugins libs/survey_parser_plugins
COPY schemas/sorting_hat_step schemas/sorting_hat_step
COPY schemas/elasticc schemas/elasticc
COPY schemas/ztf schemas/ztf 
COPY schemas/atlas schemas/atlas 

WORKDIR /app
RUN poetry install --no-root --without=dev --without=test


FROM python:3.10-slim as production
RUN pip install poetry

COPY --from=builder /app /app
COPY --from=builder /libs /libs
COPY --from=builder /schemas /schemas

COPY sorting_hat_step/sorting_hat_step /app/sorting_hat_step
COPY sorting_hat_step/scripts /app/scripts
COPY sorting_hat_step/credentials.py \
  sorting_hat_step/settings.py \
  sorting_hat_step/README.md \
  /app/

WORKDIR /app
RUN poetry install --only-root

CMD ["poetry", "run", "step"]
