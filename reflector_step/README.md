# Alerce Reflector Step

Copies one or more Kafka topics into a new one, essentially a custom made 
[mirrormaker](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=27846330). 
Used to replicate messages from an external source into the current Kafka 
cluster.

This step does nothing with the data. Custom consumers/producers completely
skip the (de)serialization stages and messages are copied just as they are.
There is no database connection.

## Environment variables

Unless noted, the following environment variables are required

## Local installation

Install required packages using:
```commandline
pip install -r requirements.txt
```

The step itself can be run with:
```commandline
python scripts/run_step.py
```

## Development and testing

Additional dependencies for testing without the deployment of the full 
infrastructure are required. these can be installed using:
```commandline
pip install -r dev-requirements.txt
```

To run all tests, use:
```commandline
pytest
```

## Previous conditions

No special conditions, only connection to Kafka.

## Version

* 1.0.0

## Libraries used

* [APF](https://github.com/alercebroker/APF)

## Environment variables

### Consumer setup

* `CONSUMER_SERVER`: Kafka host with ports, e.g., `localhost:9092`
* `CONSUMER_GROUP_ID`: Name for the consumer group, e.g., `reflector-step`
* `CONSUME_TIMEOUT`: Maximum time in seconds to wait for a message. Defaults to `10`
* `CONSUME_MESSAGES`: Number of messages to consume per operation. Defaults to `1000`
* `TOPIC_STRATEGY_FORMAT`: Format of topics that change daily, e.g., `ztf_%s_programid1` or `ztf_%s_programid1,ztf_%s_programid3`. The `{}` will be replaced by the date formatted as `%Y%m%d`, set to change every day at 23:00 UTC
* `CONSUMER_TOPICS`: List of topics to consume as a string separated by commas, e.g., `topic` or `topic1,topic2`

Note that one of `TOPIC_STRATEGY_FORMAT` or `CONSUMER_TOPICS` *must* be set. 
If both are set, then `CONSUMER_TOPICS` will be ignored.

### Producer setup

* `PRODUCER_SERVER`: Kafka host with ports, e.g., `localhost:9092`
* `PRODUCER_TOPIC`: Topic to which messages will be produced, e.g., `topic`. This is optional, if not provided, it will produce with the same topic as the incoming message.

### Step metadata

* `STEP_VERSION`: Current version of the step, e.g., `1.0.0`. Defaults to `dev`
* `STEP_ID`: Unique identifier for the step. Defaults to `reflector`
* `STEP_NAME`: Name of the step. Defaults to `reflector`
* `STEP_COMMENTS`: Comments on the specific version of the step

## Stream

This step requires a consumer and a producer. Schemas are ignored and copied 
as they are from the source.

## Build docker image

To run this step you must first build the Docker image:
```commandline
docker build -t alerce_reflector_step:version .
```

## Run step

### Run Docker container

To run the Docker container all environment variables must be set:
```commandline
docker run --name alerce_reflector_step -e CONSUMER_SERVER=host:port -e [... all env ...] -d alerce_reflector_step:version
```

### Run docker-compose

Alternatively, the environment variables can be edited in the file `docker-compose.yml`.
To run one container, use:
```commandline
docker-compose up -d
```

In order to scale up the number of containers, use:
```commandline
docker-compose up -d --scale alerce_reflector_step=n
```

**Note:** To stop all containers, use: `docker-compose down`.

### Run released image

For each release an image is uploaded to ghcr.io that can be used instead of 
building your own. To do that, replace the commands above with:
```commandline
docker pull ghcr.io/alercebroker/alerce_reflector_step:latest
```
