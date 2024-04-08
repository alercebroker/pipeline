FROM python:3.10 as python-base
MAINTAINER ALeRCE
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONFAULTHANDLER=1 \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VIRTUALENVS_IN_PROJECT=true \
  POETRY_NO_INTERACTION=1
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y --no-install-recommends git build-essential && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

FROM python-base as builder
WORKDIR /app/
RUN pip install poetry
COPY feature_step/poetry.lock feature_step/pyproject.toml /app/
COPY feature_step/README.md /app/README.md
COPY P4J /P4J
COPY mhps /mhps
COPY lc_classifier /lc_classifier
COPY libs/apf /libs/apf
COPY libs/test_utils /libs/test_utils
COPY schemas/feature_step /schemas/feature_step
COPY schemas/xmatch_step /schemas/xmatch_step
COPY schemas/scribe_step /schemas/scribe_step
RUN poetry run python -m pip install setuptools wheel Cython==0.29.36 numpy
RUN poetry run python -m pip install ../mhps --no-use-pep517
RUN poetry install --without=test --no-root
COPY feature_step/scripts /app/scripts
COPY feature_step/settings.py /app/settings.py
COPY feature_step/features /app/features

FROM python:3.10-slim
COPY --from=builder /app /app
COPY --from=builder /libs/apf /libs/apf
COPY --from=builder /mhps /mhps
COPY --from=builder /P4J /P4J
COPY --from=builder /lc_classifier /lc_classifier
COPY --from=builder /schemas /schemas
RUN pip install poetry
WORKDIR /app/
RUN poetry install --only-root
CMD ["poetry", "run", "python", "scripts/run_step.py"]