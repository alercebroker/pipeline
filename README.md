# LC Classifier

## Description

- Classification with [`Hierarchical RF`](https://github.com/alercebroker/late_classifier/blob/master/late_classifier/classifier/hierarchical_rf/model.py) model.
- Download model from [S3 bucket](https://assets.alerce.online/pipeline/hierarchical_rf_1.0.1/). - Version 1.0.1 of model from Light Curve Classification Paper.
- Do inference of features of light curve and store result. 
- If lack of some feature, so it isn't classify.

#### Previous steps:

- [LC Features](https://github.com/alercebroker/feature_step)

## Database interactions

### Insert/Update

- Table `Probability`
- Table `Taxonomy`

## Previous conditions

- Fields of required features.

## Version

- **0.0.1:** Use of APF. - Use of [Late Classifier Library](https://github.com/alercebroker/late_classifier).
- **0.0.2:** Added `db-plugins` instead of `apf.db`, using new `Probability` and `Taxonomy` tables.
- **1.0.0:** Update documentation and test step.
 
## Libraries used

- APF
- Numpy
- Pandas
- Scipy
- Late Classifier Library

## Environment variables

### Database setup

- `DB_HOST`: Database host for connection.
- `DB_USER`: Database user for read/write (requires these permission).
- `DB_PASSWORD`: Password of user.
- `DB_PORT`: Port connection.
- `DB_NAME`: Name of database.

### Consumer setup

- `CONSUMER_TOPICS`: Some topics. String separated by commas. e.g: `topic_one` or `topic_two,topic_three`
- `CONSUMER_SERVER`: Kafka host with port. e.g: `localhost:9092`
- `CONSUMER_GROUP_ID`: Name for consumer group. e.g: `correction`

### Producer setup

- `PRODUCER_TOPIC`: Name of output topic. e.g: `correction`
- `PRODUCER_SERVER`: Kafka host with port. e.g: `localhost:9092`

## Stream

### Input schema

- Output stream of [`Feature Step`](https://github.com/alercebroker/feature_step#output-schema).

### Output schema

```json
{
  "doc": "Late Classification",
  "name": "probabilities + features",
  "type": "record",
  "fields": [
    {
      "name": "features",
      "type": {
        "type": "record",
        "name": "features_record",
        "fields": [
          { "name": "oid", "type": "string" },
          {
            "name": "features",
            "type": {
              "type": "map",
              "values": ["float", "int", "string", "null"]
            }
          }
        ]
      }
    },
    {
      "name": "late_classification",
      "type": {
        "type": "record",
        "name": "late_record",
        "fields": [
          {
            "name": "probabilities",
            "type": {
              "type": "map",
              "values": ["float"]
            }
          },
          {
            "name": "class",
            "type": "string"
          },
          {
            "name": "hierarchical",
            "type": {
              "name": "root",
              "type": "map",
              "values": [
                {
                  "type": "map",
                  "values": "float"
                },
                {
                  "type": "map",
                  "values": {
                    "type": "map",
                    "values": "float"
                  }
                }
              ]
            }
          }
        ]
      }
    }
  ]
}
```

## Build docker image

For use this step, first you must build the image of docker. After that you can run the step for use it.

```bash
docker build -t late_classification_step:latest .
```

## Run step

### Run container of docker

You can use a `docker run` command, you must set all environment variables.

```bash
docker run --name my_step -e DB_NAME=myhost -e [... all env ...] -d late_classification_step:latest
```

### Run docker-compose

Also you can edit the environment variables in [`docker-compose.yml`](https://github.com/alercebroker/feature_step/blob/master/docker-compose.yml) file. After that use `docker-compose up` command. This run only one container.

```bash
docker-compose up -d
```

If you want scale this container, you must set a number of containers to run.

```bash
docker-compose up -d --scale late_classification=32
```

**Note:** Use `docker-compose down` for stop all containers.

## Local Installation

### Requirements

To install the required packages run

```bash
pip install -r requirements.txt
```

After that you can modify the logic of the step in [step.py](https://github.com/alercebroker/late_classification_step/blob/master/late_classification/step.py) and run

```bash
python scripts/run_step.py
```