# Alerce Rubin Stamp Classifier Step

This step is part of the ALeRCE astronomical alert broker pipeline. It processes incoming alert messages from telescopes such as the Rubin Observatory, classifies the alert stamps, and writes the results to a Kafka topic. The step is designed to be modular and easily integrated into the broader ALeRCE pipeline.

## Overview

- **Input:** Reads Avro-encoded alert messages from a Kafka topic (e.g., `lsst`).
- **Processing:** Applies the Rubin stamp classification logic to each alert.
- **Output:** Writes classification results to another Kafka topic (e.g., `rubin_stamp_classifier`).
- **Optional:** May interact with a database for storing results or metadata.

## Local Installation

1. **Configuration:**
   
   Create a YAML file named `local_config.yaml` with content similar to:
   ```yaml
   LOGGING_LEVEL: DEBUG
   STEP_CONFIG:
     CONSUMER_CONFIG:
       CLASS: "rubin_stamp_classifier_step.utils.LsstKafkaConsumer"
       TOPICS: ["lsst"]
       PARAMS:
         bootstrap.servers: localhost:9092
         group.id: rubin-stamp-classifier
         auto.offset.reset: earliest
       consume.timeout: 5
       consume.messages: 16
       SCHEMA_PATH: "/schemas/surveys/lsst/v7_4_alert.avsc"
     PRODUCER_CONFIG:
       CLASS: "apf.producers.kafka.KafkaSchemalessProducer"
       TOPIC: rubin_stamp_classifier
       PARAMS:
         bootstrap.servers: localhost:9092
       SCHEMA_PATH: "schemas/rubin_stamp_classifier_step/output.avsc"
     DB_CONFIG:
       USER: postgres
       PASSWORD: postgres
       HOST: localhost
       PORT: 5432
       DB_NAME: postgres
       SCHEMA: public
     MODEL_VERSION: "1.0.0"
     MODEL_CONFIG:
       # MODEL_PATH: "/path/to/local/model"
       MODEL_PATH: "https://download.my.model/model.zip"
     FEATURE_FLAGS:
       USE_PROFILING: false
       PROMETHEUS: false
   ```
   Adjust parameters as needed for your environment.

2. **Set the config path:**
   ```bash
   export CONFIG_YAML_PATH=/path/to/local_config.yaml
   ```

3. **Install dependencies:**
   It is recommended to use Poetry:
   ```bash
   poetry install
   ```

4. **Run the step:**
   ```bash
   poetry run step
   ```

## Integration Testing

To run the integration tests, set rubin_stamp_classifier_step as the current directory.
Then set the environment variable `TEST_RUBIN_STAMP_CLASSIFIER_STEP_MODEL_PATH` with 
the path of the directory containing the model files. You can also use a URL to a zip file containing the model.

```bash
export TEST_RUBIN_STAMP_CLASSIFIER_STEP_MODEL_PATH=/path/to/model
# or
export TEST_RUBIN_STAMP_CLASSIFIER_STEP_MODEL_PATH=https://download.my.model/model.zip
```
Then run the tests with:

```bash
poetry run pytest -m integration
```

## Helm Deployment

To deploy this step using Helm, use the provided `values.yaml` as a template. Adjust parameters for your Kafka setup and topics.

```yaml
namespace: pipeline-rubin-stamp-classifier-step
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 1
    memory: 5Gi
autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 8
  targetCPUUtilizationPercentage: 60
image:
  repository: ghcr.io/alercebroker/rubin_stamp_classifier_step
  tag: tag_version
configYaml:
  enabled: true
  LOGGING_LEVEL: DEBUG
  STEP_CONFIG:
    CONSUMER_CONFIG:
      CLASS: "rubin_stamp_classifier_step.utils.LsstKafkaConsumer"
      TOPICS: ["lsst"]
      PARAMS:
        bootstrap.servers: localhost:9092
        group.id: rubin-stamp-classifier
        auto.offset.reset: earliest
      consume.timeout: 5
      consume.messages: 16
      SCHEMA_PATH: "/schemas/surveys/lsst/v7_4_alert.avsc"
    PRODUCER_CONFIG:
      CLASS: "apf.producers.kafka.KafkaSchemalessProducer"
      TOPIC: rubin_stamp_classifier
      PARAMS:
        bootstrap.servers: localhost:9092
      SCHEMA_PATH: "schemas/rubin_stamp_classifier_step/output.avsc"
    DB_CONFIG:
      USER: postgres
      PASSWORD: postgres
      HOST: localhost
      PORT: 5432
      DB_NAME: postgres
      SCHEMA: public
    MODEL_VERSION: "1.0.0"
    MODEL_CONFIG:
      # MODEL_PATH: "/path/to/local/model"
      MODEL_PATH: "https://download.my.model/model.zip"
    FEATURE_FLAGS:
      USE_PROFILING: false
      PROMETHEUS: false
```

Deploy with:
```bash
helm install pipeline-rubin-stamp-classifier-step alerce-pipeline/rubin-stamp-classifier-step -f values.yaml
```

Upgrade with:
```bash
helm upgrade pipeline-rubin-stamp-classifier-step alerce-pipeline/rubin-stamp-classifier-step -f values.yaml
```

Uninstall with:
```bash
helm uninstall pipeline-rubin-stamp-classifier-step
```

## Development

- Source code: `rubin_stamp_classifier_step/`
- Tests: `tests/`
- Schemas: `schemas/surveys/lsst/`

