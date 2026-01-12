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
COPY rubin_stamp_classifier_step/pyproject.toml \
    /app/
COPY libs/apf /libs/apf
COPY libs/db-plugins-multisurvey /libs/db-plugins-multisurvey
COPY alerce_classifiers /alerce_classifiers
COPY schemas /schemas
COPY schemas /schemas
WORKDIR /app
RUN poetry install --no-root --without=dev --with=rubin


FROM python:3.10-slim as production
RUN pip install poetry
COPY --from=builder /app /app
COPY --from=builder /libs/apf /libs/apf
COPY --from=builder /libs/db-plugins-multisurvey /libs/db-plugins-multisurvey
COPY --from=builder /alerce_classifiers /alerce_classifiers
COPY --from=builder /schemas /schemas
COPY rubin_stamp_classifier_step/scripts /app/scripts
COPY rubin_stamp_classifier_step/README.md /app/README.md
COPY rubin_stamp_classifier_step/rubin_stamp_classifier_step /app/rubin_stamp_classifier_step
WORKDIR /app/
RUN poetry install --only-root
CMD ["poetry", "run", "python", "scripts/run_step.py"]