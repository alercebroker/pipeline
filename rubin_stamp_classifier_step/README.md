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
   LOGGING_LEVEL: INFO
   STEP_CONFIG:
     CONSUMER_CONFIG:
       CLASS: "rubin_stamp_classifier_step.utils.LsstKafkaConsumer"
       TOPICS: ["lsst"]
       PARAMS:
         bootstrap.servers: localhost:9092
         group.id: rubin-stamp-classifier
         auto.offset.reset: earliest
     PRODUCER_CONFIG:
       CLASS: "rubin_stamp_classifier_step.utils.RawKafkaProducer"
       TOPIC: rubin_stamp_classifier
       PARAMS:
         bootstrap.servers: localhost:9092
         acks: "all"
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

Integration tests use Docker Compose to spin up required services (Kafka, Postgres). Tests populate the input Kafka topic with Avro messages and verify the output topic for correct classification results.

- Test data: `tests/integration/data/avro_messages/`
- Example test: `tests/integration/test_kafka_output.py`

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
  LOGGING_LEVEL: INFO
  STEP_CONFIG:
    CONSUMER_CONFIG:
      CLASS: "rubin_stamp_classifier_step.utils.LsstKafkaConsumer"
      TOPICS: ["lsst"]
      PARAMS:
        bootstrap.servers: kafka:9092
        group.id: rubin-stamp-classifier
        auto.offset.reset: earliest
    PRODUCER_CONFIG:
      CLASS: "rubin_stamp_classifier_step.utils.RawKafkaProducer"
      TOPIC: rubin_stamp_classifier
      PARAMS:
        bootstrap.servers: kafka:9092
        acks: "all"
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

