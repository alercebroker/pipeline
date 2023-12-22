FROM python:3.9-slim as builder
MAINTAINER ALeRCE
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONFAULTHANDLER=1 \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VIRTUALENVS_IN_PROJECT=true \
  POETRY_NO_INTERACTION=1
WORKDIR /app/
RUN pip install poetry
COPY ./alert_archiving_step/poetry.lock ./alert_archiving_step/pyproject.toml /app/
RUN poetry install --no-interaction --without=dev --no-root

FROM python:3.9-slim
RUN pip install poetry
WORKDIR /app
COPY --from=builder /app /app
COPY ./alert_archiving_step/README.md /app/
COPY ./alert_archiving_step/settings.py /app/
COPY ./alert_archiving_step/scripts /app/scripts
COPY ./alert_archiving_step/alert_archiving_step /app/alert_archiving_step
RUN poetry install --no-interaction --only-root
CMD ["poetry", "run", "run-step"]
