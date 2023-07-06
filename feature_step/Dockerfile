FROM python:3.10-slim as builder
MAINTAINER ALeRCE
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN apt-get update && \
  apt-get install -y --no-install-recommends git build-essential && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
WORKDIR /app/
RUN pip install poetry
COPY ./poetry.lock ./pyproject.toml /app/
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --without=dev --no-root


FROM builder
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
COPY ./features /app/features
COPY ./scripts /app/scripts
COPY ./README.md /app/
RUN poetry install --no-interaction --only-root
CMD ["poetry", "run", "python", "scripts/run_step.py"]
