FROM python:3.10-slim as python-base
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
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install poetry
COPY lc_classification_multisurvey_step/pyproject.toml /app/
# Requires apf >= commit 49a95d98b, which added GenericStep._flush_producers.
# step.py's produce_scribe does no manual flush and relies on the framework
# flushing every producer before the offset commit; on an older apf, scribe
# writes would be committed over and lost silently on pod deletion.
COPY libs/apf /libs/apf
COPY libs/db-plugins-multisurvey /libs/db-plugins-multisurvey
COPY alerce_classifiers /alerce_classifiers
COPY schemas/scribe_step /schemas/scribe_step
COPY schemas/lc_classification_step /schemas/lc_classification_step
WORKDIR /app
RUN poetry install --no-root


FROM python:3.10-slim as production
RUN pip install poetry
COPY --from=builder /app /app
COPY --from=builder /libs/apf /libs/apf
COPY --from=builder /libs/db-plugins-multisurvey /libs/db-plugins-multisurvey
COPY --from=builder /alerce_classifiers /alerce_classifiers
COPY --from=builder /schemas /schemas
COPY lc_classification_multisurvey_step/scripts /app/scripts
COPY lc_classification_multisurvey_step/README.md /app/README.md
COPY lc_classification_multisurvey_step/settings.py /app/settings.py
COPY lc_classification_multisurvey_step/models_settings.py /app/models_settings.py
COPY lc_classification_multisurvey_step/lc_classification_multisurvey_step /app/lc_classification_multisurvey_step

WORKDIR /app/
RUN poetry install --only-root
CMD ["poetry", "run", "python", "scripts/run_step.py"]
