FROM python:3.10-slim as builder
MAINTAINER ALeRCE
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
RUN apt-get update && \
  apt-get install -y --no-install-recommends git build-essential && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
WORKDIR /app/
RUN pip install poetry
RUN pip install numpy
COPY ./poetry.lock ./pyproject.toml /app/
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --without=dev --without=test --no-root
COPY ./features /app/features
COPY ./scripts /app/scripts
COPY ./settings.py /app/settings.py
COPY ./README.md /app/README.md
COPY ./schema.py /app/schema.py
COPY ./scribe_schema.avsc /app/scribe_schema.avsc



FROM builder
RUN poetry install --no-interaction --only-root
CMD ["poetry", "run", "python", "scripts/run_step.py"]
