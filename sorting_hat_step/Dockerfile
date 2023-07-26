FROM python:3.8-slim as python-base
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
WORKDIR /app
COPY ./poetry.lock ./pyproject.toml ./
RUN poetry install --no-root --without=dev --without=test

FROM python:3.8-slim as production
RUN pip install poetry
COPY --from=builder /app /app
WORKDIR /app
COPY ./credentials.py /app/credentials.py
COPY ./settings.py /app/settings.py
COPY ./README.md /app/README.md
COPY ./schemas /app/schemas
COPY ./scripts /app/scripts
COPY ./sorting_hat_step /app/sorting_hat_step
RUN poetry install --only-root
CMD ["poetry", "run", "step"]
