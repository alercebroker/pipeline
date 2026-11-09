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

## LSST: surviving Rubin schema changes

Rubin publishes alerts in Confluent wire format: a magic byte, a 4-byte schema id and
the Avro body. The id is built from the schema version (`lsst.v11_2` -> `1102`).
`reflector_step.utils.LsstMultiVersionRegistryConsumer` reads that id and decodes each
alert with the matching schema, then upgrades older alerts to `SCHEMA_PATH`, so the
output topic always carries a single schema version. This lets the reflector be
deployed *before* Rubin switches schema, and keeps it working if both versions are
mixed on the topic. An alert with an unknown schema id makes the step fail instead of
being decoded with the wrong schema.

```yaml
CONSUMER_CONFIG:
  CLASS: reflector_step.utils.LsstMultiVersionRegistryConsumer
  SCHEMA_PATH: /schemas/surveys/lsst_v11.2/lsst.v11_2.alert.avsc  # output version
  PREVIOUS_SCHEMA_PATHS:                                          # also accepted
    - /schemas/surveys/lsst_v11.1/lsst.v11_1.alert.avsc
PRODUCER_CONFIG:
  CLASS: apf.producers.KafkaSchemalessProducer
  SCHEMA_PATH: /schemas/surveys/lsst_v11.2/lsst.v11_2.alert.avsc  # same as consumer
```

Fields Rubin adds without an Avro default need a value for older alerts; it is set in
`LsstMultiVersionRegistryConsumer.UPGRADE_DEFAULTS` (v11.2: `templateInstrument="UNKNOWN"`,
`templateVersion=0`, `mixedInstruments=False`).

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