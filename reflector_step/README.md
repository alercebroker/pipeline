# Alerce Reflector Step

Copies one or more Kafka topics into a new one, essentially a custom-made 
[mirrormaker](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=27846330). 
Used to replicate messages from an external source into the current Kafka 
cluster.

This step does nothing with the data. Custom consumers/producers completely
skip the (de)serialization stages and messages are copied just as they are.
There is no database connection.

## Local installation

Create a yaml file named `config.yaml` with the following content:
```yaml
LOGGING_LEVEL: INFO
STEP_CONFIG:
  FEATURE_FLAGS:
    USE_PROFILING: false
    PROMETHEUS: false
  FILTERS_CONFIG:
    ENABLE: true
    MODULUS: 100
    REMAINDER: 1
  CONSUMER_CONFIG:
    CLASS: "reflector_step.utils.RawKafkaConsumer"
    TOPICS: ["topic_1", "topic_2"]
    PARAMS:
      bootstrap.servers: remote-kafka.server:port
      group.id: consumer-group-id
      security.protocol: SASL_SSL
      sasl.mechanisms: SCRAM-SHA-512
      sasl.username: username
      sasl.password: password
      message.max.bytes: 10000000
    consume.messages: 100
    consume.timeout: 20
  keep_original_timestamp: false
  use_message_topic: false
  PRODUCER_CONFIG:
    CLASS: "reflector_step.utils.RawKafkaProducer"
    TOPIC: output_topic
    PARAMS:
      bootstrap.servers: local-kafka.server:port
      acks: "all"
      security.protocol: SASL_SSL
      sasl.mechanisms: SCRAM-SHA-512
      sasl.username: username
      sasl.password: password

```
Adjust the parameters to your needs.
Set an environment variable `CONFIG_YAML_PATH` to the location of the `config.yaml` file, e.g.:
```commandline
export CONFIG_YAML_PATH=/path/to/config.yaml
```

Then, install the required Python packages. It is recommended to use poetry.

```commandline
poetry install
```

The step itself can be run with:
```commandline
poetry run step
```

## LSST: Rubin schema version check

Rubin publishes alerts in Confluent wire format: a magic byte, a 4-byte schema id and
the Avro body. The id is built from the schema version (`lsst.v11_2` -> `1102`).
`reflector_step.utils.LsstRegistryConsumer` only accepts alerts whose id matches
`SCHEMA_PATH`. Any other id (an older or a newer Rubin schema), a missing header, or
bytes left over after decoding make the step fail with an error naming the topic,
partition, offset and both versions, instead of decoding the alert with the wrong
schema. When Rubin changes schema, the reflector stops at the first new alert until
`SCHEMA_PATH` and the pipeline are updated.

```yaml
CONSUMER_CONFIG:
  CLASS: reflector_step.utils.LsstRegistryConsumer
  SCHEMA_PATH: /schemas/surveys/lsst_v11.2/lsst.v11_2.alert.avsc  # only accepted version
PRODUCER_CONFIG:
  CLASS: apf.producers.KafkaSchemalessProducer
  SCHEMA_PATH: /schemas/surveys/lsst_v11.2/lsst.v11_2.alert.avsc  # same as consumer
```

## Helm deployment

To deploy this step using Helm, you can use the provided `values.yaml` file as a template. Adjust the parameters according to your Kafka setup and the topics you want to mirror.

```yaml
namespace: pipeline-reflector-step
affinity:
  nodeAffinity: null
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
  repository: ghcr.io/alercebroker/reflector_step
  tag: tag_version
imageCredentials:
  password: password
  registry: ghcr.io
  username: username
configYaml:
    enabled: true
    LOGGING_LEVEL: INFO
    STEP_CONFIG:
      FEATURE_FLAGS:
        USE_PROFILING: false
        PROMETHEUS: false
      CONSUMER_CONFIG:
        CLASS: "reflector_step.utils.RawKafkaConsumer"
        TOPICS: ["topic_1", "topic_2"]
        PARAMS:
          bootstrap.servers: remote-kafka.server:port
          group.id: consumer-group-id
          security.protocol: SASL_SSL
          sasl.mechanisms: SCRAM-SHA-512
          sasl.username: username
          sasl.password: password
          message.max.bytes: 10000000
        consume.messages: 100
        consume.timeout: 20
      keep_original_timestamp: false
      use_message_topic: false
      PRODUCER_CONFIG:
        CLASS: "reflector_step.utils.RawKafkaProducer"
        TOPIC: output_topic
        PARAMS:
          bootstrap.servers: local-kafka.server:port
          acks: "all"
          security.protocol: SASL_SSL
          sasl.mechanisms: SCRAM-SHA-512
          sasl.username: username
          sasl.password: password
```

To deploy the step, run the following command in the directory containing the `values.yaml` file:

```bash
helm install pipeline-reflector-step alerce-pipeline/reflector-step -f values.yaml
```

To upgrade the deployment with new configurations, use:

```bash
helm upgrade pipeline-reflector-step alerce-pipeline/reflector-step -f values.yaml
```

To uninstall the step, run:

```bash
helm uninstall pipeline-reflector-step
```