# Early Classification Step

## Description

Process ZTF Stream using the [ALeRCE Stamp classifier](https://github.com/alercebroker/stamp_classifier) and
store probabilities to the database.

#### Previous steps: 
- None

#### Next steps:
- None

## Database interactions
Save step metadata and probabilities. The following config is needed

Tables:
- step
- probability

```python
DB_CONFIG = {
    "SQL": {
        "ENGINE": os.environ["DB_ENGINE"],
        "HOST": os.environ["DB_HOST"],
        "USER": os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "PORT": int(os.environ["DB_PORT"]),
        "DB_NAME": os.environ["DB_NAME"],
    }
}

STEP_METADATA = {
    "STEP_VERSION": os.getenv("STEP_VERSION", "dev"),
    "STEP_ID": os.getenv("STEP_ID", "features"),
    "STEP_NAME": os.getenv("STEP_NAME", "features"),
    "STEP_COMMENTS": os.getenv("STEP_COMMENTS", ""),
    "FEATURE_VERSION": os.getenv("FEATURE_VERSION", "dev"),
}

STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "STEP_METADATA": STEP_METADATA,
}
```

## Previous conditions

- None

## Version
- **1.0.0:** 
	- Use of APF.
    - Use of db-plugins
	- Use of stamp classifier model.

## Libraries used
- APF
- db-plugins
- stamp_classifier (imported as zip from alerce repository)

## Environment variables

### Consumer setup

- `CONSUMER_TOPICS`: Some topics. String separated by commas. e.g: `topic_one` or `topic_two,topic_three`
- `CONSUMER_SERVER`: Kafka host with port. e.g: `localhost:9092`
- `CONSUMER_GROUP_ID`: Name for consumer group. e.g: `correction`

### DB setup
- `DB_ENGINE`: Database engine used is `postgresql`
- `DB_HOST`: Database host name or ip. e.g: `localhost`
- `DB_USER`: Database user name. e.g: `postgres`
- `DB_PASSWORD`: Database password. e.g: `postgres`
- `DB_PORT`: Database port. e.g: `5432`
- `DB_NAME`: Database name: e.g: `postgres`

### Step metadata
- `STEP_VERSION`: Current version of the step. e.g: `1.0.0`
- `STEP_ID`: Unique identifier for the step. e.g: `S3`
- `STEP_NAME`: Name of the step. e.g: `S3`
- `STEP_COMMENTS`: Comments of the specific version.


## Stream

This step require only consumer.

### Input schema
- [Documentation of ZTF Avro schema.](https://zwickytransientfacility.github.io/ztf-avro-alert/schema.html)
- [Avsc files](https://github.com/ZwickyTransientFacility/ztf-avro-alert/tree/master/schema)

### Output schema
- None

## Run step

### Run container of docker
You can use a `docker run` command, you must set all environment variables.
```bash
docker run --name my_step -e DB_HOST=myhost -e [... all env ...] -d stamp_step:version
```

### Run docker-compose
Also you can edit the environment variables in [`docker-compose.yml`](https://github.com/alercebroker/early_classification_step/blob/main/docker-compose.yml) file. After that use `docker-compose up` command. This run only one container.

```bash
docker-compose up -d
```

If you want scale this container, you must set a number of containers to run.

```bash
docker-compose up -d --scale early_classification_step=32
```

**Note:** Use `docker-compose down` for stop all containers.

### Run the released image
For each release an image is uploaded to ghcr.io that you can use instead of building your own. To do that replace `docker-compose.yml` or the `docker run` command with this image:

```bash
docker pull ghcr.io/alercebroker/early_classification_step:latest
```
## Local Installation

### Requirements

To install the required packages run
```python
  pip install -r requirements.txt
```


### Downloading and installing the model

We are using the zip model from `alercebroker/stamp_classifier` releases.
[model.zip (v1.0.1)](https://github.com/alercebroker/stamp_classifier/releases/download/1.0.0/model.zip)


The Early Classification Step expect the model inside the `model` folder.
```bash
  unzip model.zip -c model
```

Then we need to install the model's required packages.
```python
  pip install -r model/requirements.txt
```
