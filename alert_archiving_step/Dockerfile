FROM python:3.9-slim as builder
MAINTAINER ALeRCE
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app/
RUN pip install poetry
COPY ./poetry.lock ./pyproject.toml /app/
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --without=dev --no-root

FROM builder
COPY ./alert_archiving_step /app/alert_archiving_step
COPY ./scripts /app/scripts
COPY ./README.md /app/
RUN poetry install --no-interaction --only-root
CMD ["poetry", "run", "run-step"]