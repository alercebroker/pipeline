# LC Features 

## Description

- Only for light curves in r-band and g-band.
- Use of [DetectionsV2PreprocessorZTF](https://github.com/alercebroker/late_classifier/blob/master/late_classifier/features/preprocess/preprocess_ztf.py#L78):
	- Drop duplicates.
	- Discard invalid value detections.
	- Discard noisy detections.
- Get features from [CustomHierarchicalExtractor](https://github.com/alercebroker/late_classifier/blob/master/late_classifier/features/custom/custom_hierarchical.py#L16):
	- `GalacticCoordinatesExtractor`
	- `SGScoreExtractor`
	- `ColorFeatureExtractor`
	- `RealBogusExtractor`
	- `MHPSExtractor`
	- `IQRExtractor`
	- `TurboFatsFeatureExtractor`
	- `SupernovaeNonDetectionFeatureExtractor` (this includes `SupernovaeDetectionFeatureExtractor`)
- Return a DataFrame of ~150 columns by object.
- In `settings.py` you must to set the version of features. e.g: `FEATURE_VERSION = "v0.1"`

#### Previous steps: 
- [LC Correction](https://github.com/alercebroker/correction_step)

#### Next steps:
- [LC Classifier](https://github.com/alercebroker/late_classification_step)

## Database interactions

### Insert/Update
- Table `features_object`: insert a json with version of features.

## Previous conditions

- Have installed  [Late Classifier Library](https://github.com/alercebroker/late_classifier) without problems. Maybe you could have problems with:
	- `Turbo-fats`
	- `P4J`
	- `MHPS`
	
**Note:** Go to each repository for instructions to install.

## Version
- **0.0.1:** 
	- Use of APF.
	- Use of [Late Classifier Library](https://github.com/alercebroker/late_classifier).

## Libraries used
- APF
- Numpy
- Pandas
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

### Elasticsearch setup
- `ES_PREFIX`: Enables the indexing of term prefixes to speed up prefix searches. e.g: `ztf_pipeline`
- `ES_NETWORK_HOST`: Elasticsearch host.
- `ES_NETWORK_PORT`: Elasticsearch port.

## Stream

This step require only consumer.

### Input schema
- Output stream of  [`Correction Step`](https://github.com/alercebroker/correction_step#output-schema).

### Output schema
```json
  "doc": "Features",
  "name": "features",
  "type": "record",
  "fields": [
    {
      "name": "oid",
      "type": "string"
    },
    {
      "name": "features",
      "type": {
        "type": "map",
        "values": [
          "float",
          "int",
          "string",
          "null"
        ]
      }
    }
  ]
}
```

## Build docker image
For use this step, first you must build the image of docker. After that you can run the step for use it.

```bash
docker build -t features_step:0.0.1 . 
```

## Run step

### Run container of docker
You can use a `docker run` command, you must set all environment variables.
```bash
docker run --name my_features_step -e DB_NAME=myhost -e [... all env ...] -d features_step:0.0.1
```

### Run docker-compose
Also you can edit the environment variables in [`docker-compose.yml`](https://github.com/alercebroker/feature_step/blob/master/docker-compose.yml) file. After that use `docker-compose up` command. This run only one container.

```bash
docker-compose up -d
```

If you want scale this container, you must set a number of containers to run.

```bash
docker-compose up -d --scale features=32
```

**Note:** Use `docker-compose down` for stop all containers.
