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
COPY stamp_classifier_2025_multisurvey_step/pyproject.toml \
    /app/
COPY libs/apf /libs/apf
COPY libs/db-plugins-multisurvey libs/db-plugins-multisurvey
COPY alerce_classifiers /alerce_classifiers
COPY schemas/stamp_classifier_2025_multisurvey_step /schemas/stamp_classifier_2025_multisurvey_step
COPY schemas/ztf /schemas/ztf
COPY schemas/scribe_step /schemas/scribe_step
WORKDIR /app
COPY libs/idmapper /libs/idmapper
RUN poetry install --no-root 


FROM python:3.10-slim as production
RUN pip install poetry
COPY --from=builder /app /app
COPY --from=builder /libs/apf /libs/apf
COPY --from=builder /libs/db-plugins-multisurvey /libs/db-plugins-multisurvey
COPY --from=builder /alerce_classifiers /alerce_classifiers
COPY --from=builder /schemas /schemas
COPY --from=builder /libs/idmapper /libs/idmapper
COPY stamp_classifier_2025_multisurvey_step/scripts /app/scripts
COPY stamp_classifier_2025_multisurvey_step/README.md /app/README.md
COPY stamp_classifier_2025_multisurvey_step/settings.py /app/settings.py
COPY stamp_classifier_2025_multisurvey_step/models_settings.py /app/models_settings.py
COPY stamp_classifier_2025_multisurvey_step/stamp_classifier_2025_multisurvey_step /app/stamp_classifier_2025_multisurvey_step
COPY --from=builder /libs/idmapper /libs/idmapper

WORKDIR /app/
RUN poetry install --only-root
CMD ["poetry", "run", "python", "scripts/run_step.py"]  