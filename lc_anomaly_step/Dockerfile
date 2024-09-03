FROM python:3.10-slim as python-base
LABEL org.opencontainers.image.authors="ALeRCE"
ARG model
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
    apt-get install -y --no-install-recommends git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install poetry
COPY lc_classification_step/poetry.lock \
    lc_classification_step/pyproject.toml \
    /app/
COPY libs/apf /libs/apf
COPY alerce_classifiers /alerce_classifiers
COPY lc_classifier /lc_classifier
COPY mhps /mhps
COPY P4J /P4J
COPY schemas/lc_classification_step /schemas/lc_classification_step
COPY schemas/feature_step /schemas/feature_step
COPY schemas/scribe_step /schemas/scribe_step
WORKDIR /app
RUN poetry run python -m pip install setuptools wheel Cython==0.29.36 numpy
RUN poetry run python -m pip install ../mhps --no-use-pep517
RUN poetry install --no-root --without=dev --without=test --with=${model}


FROM python:3.10-slim as production
RUN pip install poetry
COPY --from=builder /app /app
COPY --from=builder /libs/apf /libs/apf
COPY --from=builder /alerce_classifiers /alerce_classifiers
COPY --from=builder /lc_classifier /lc_classifier
COPY --from=builder /mhps /mhps
COPY --from=builder /P4J /P4J
COPY --from=builder /schemas /schemas
COPY lc_classification_step/scripts /app/scripts
COPY lc_classification_step/README.md /app/README.md
COPY lc_classification_step/settings.py /app/settings.py
COPY lc_classification_step/models_settings.py /app/models_settings.py
COPY lc_classification_step/lc_classification /app/lc_classification
WORKDIR /app/
RUN poetry install --only-root
CMD ["poetry", "run", "step"]
