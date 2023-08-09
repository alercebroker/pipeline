FROM python:3.9-slim as builder
MAINTAINER ALeRCE
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
WORKDIR /app/
RUN pip install poetry
COPY ./poetry.lock ./pyproject.toml /app/
RUN poetry install --no-interaction --without=dev --no-root


FROM python:3.9-slim
RUN pip install poetry
COPY --from=builder /app /app
COPY ./scripts /app/scripts
COPY ./README.md /app/README.md
COPY ./settings.py /app/settings.py
COPY ./reflector_step /app/reflector_step
WORKDIR /app
RUN poetry install --no-interaction --only-root
CMD ["poetry", "run", "run-step"]
