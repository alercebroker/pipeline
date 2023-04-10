FROM python:3.10-slim
MAINTAINER ALeRCE
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/

RUN pip install --no-cache-dir poetry
COPY ./poetry.lock ./pyproject.toml /app/
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-cache -E apf --without=dev --no-root

COPY ./correction /app/correction
COPY ./README.md /app/
RUN poetry install --no-interaction --no-cache -E apf --only-root

CMD ["poetry", "run", "run-step"]