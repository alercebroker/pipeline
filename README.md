# LC Features 

## Description

- Only for light curves in r-band and g-band.
- Use of [DetectionsV2PreprocessorZTF](https://github.com/alercebroker/late_classifier/blob/master/late_classifier/features/preprocess/preprocess_ztf.py#L78):
	- Drop duplicates.
	- Discard invalid value detections.
	- Discard noisy detections.
- Get features from [CustomStreamHierarchicalExtractor](https://github.com/alercebroker/late_classifier/blob/master/late_classifier/features/custom/custom_hierarchical.py#L121):
	GalacticCoordinatesExtractor(),
    - `StreamSGScoreExtractor()`
    - `ColorFeatureExtractor()`
    - `RealBogusExtractor()`
    - `MHPSExtractor()`
    - `IQRExtractor()`
    - `TurboFatsFeatureExtractor()`
    - `SupernovaeDetectionAndNonDetectionFeatureExtractor()`
    - `SNParametricModelExtractor()`
    - `WiseStreamExtractor()`
    - `PeriodExtractor(bands=bands)`
    - `PowerRateExtractor()`
    - `FoldedKimExtractor()`
    - `HarmonicsExtractor()`
- Return a DataFrame of ~150 columns by object.

#### Previous steps: 
- [LC Correction](https://github.com/alercebroker/correction_step)

#### Next steps:
- [LC Classifier](https://github.com/alercebroker/late_classification_step)

## Database interactions

### Insert/Update
- Table `feature_version`: insert version of features and step version.
- Table `object`: insert object information
- Table `feature`: insert each feature and value
- Table `step`: insert step metadata

## Previous conditions

- Have installed  [Late Classifier Library](https://github.com/alercebroker/late_classifier) without problems. 

## Version
- **1.0.0:** 
	- Use of APF.
    - Use of db-plugins
	- Use of [Late Classifier Library](https://github.com/alercebroker/late_classifier).

## Libraries used
- apf-base
- numpy
- pandas
- late_classifier
- mhps
- db-plugins
- turbo-fats
- p4j

## Environment variables

### Database setup

- `ENGINE`: Database engine. currently using postgres
- `DB_HOST`: Database host for connection.
- `DB_USER`: Database user for read/write (requires these permission).
- `DB_PASSWORD`: Password of user.
- `DB_PORT`: Port connection.
- `DB_NAME`: Name of database.

### Consumer setup

- `CONSUMER_TOPICS`: Some topics. String separated by commas. e.g: `topic_one` or `topic_two,topic_three`
- `CONSUMER_SERVER`: Kafka host with port. e.g: `localhost:9092`
- `CONSUMER_GROUP_ID`: Name for consumer group. e.g: `correction`
- `ENABLE_PARTITION_EOF`: En process when there are no more messages. Default: False

### Producer setup

- `PRODUCER_TOPIC`: Name of output topic. e.g: `correction`
- `PRODUCER_SERVER`: Kafka host with port. e.g: `localhost:9092`

### Metrics setup

- `METRICS_HOST`: Kafka host for storing metrics
- `METRICS_TOPIC`: Name of the topic to store metrics

### Step metadata
- `STEP_VERSION`: Current version of the step. e.g: `1.0.0`
- `STEP_ID`: Unique identifier for the step. e.g: `S3`
- `STEP_NAME`: Name of the step. e.g: `S3`
- `STEP_COMMENTS`: Comments of the specific version.
- `FEATURE_VERSION`: Version of the features used in the step

## Stream

This step require consumer and producer.

### Input schema
- Output stream of  [`Correction Step`](https://github.com/alercebroker/correction_step#schema.py).

### Output schema
```json
{
    "doc": "Light curve",
    "name": "light_curve",
    "type": "record",
    "fields": [
        {"name": "oid", "type": "string"},
        {"name": "candid", "type": "long"},
        {"name": "detections", "type": DETECTIONS},
        {
            "name": "non_detections",
            "type": NON_DETECTIONS,
        },
        {"name": "xmatches", "type": [XMATCH, "null"], "default": "null"},
        {"name": "fid", "type": "int"},
        {
            "name": "metadata",
            "type": METADATA,
        },
        {"name": "preprocess_step_id", "type": "string"},
        {"name": "preprocess_step_version", "type": "string"},
    ],
}
```
#### Detections
```json
{
    "type": "array",
    "items": {
        "name": "detections_record",
        "type": "record",
        "fields": [
            {"name": "oid", "type": "string"},
            {"name": "candid", "type": "long"},
            {"name": "mjd", "type": "float"},
            {"name": "fid", "type": "int"},
            {"name": "pid", "type": "float"},
            {"name": "diffmaglim", "type": "float"},
            {"name": "isdiffpos", "type": "int"},
            {"name": "nid", "type": "int"},
            {"name": "ra", "type": "float"},
            {"name": "dec", "type": "float"},
            {"name": "magpsf", "type": "float"},
            {"name": "sigmapsf", "type": "float"},
            {"name": "magap", "type": "float"},
            {"name": "sigmagap", "type": "float"},
            {"name": "distnr", "type": "float"},
            {"name": "rb", "type": "float"},
            {"name": "rbversion", "type": ["string", "null"]},
            {"name": "drb", "type": ["float", "null"]},
            {"name": "drbversion", "type": ["string", "null"]},
            {"name": "magapbig", "type": "float"},
            {"name": "sigmagapbig", "type": "float"},
            {"name": "rfid", "type": ["int", "null"]},
            {"name": "magpsf_corr", "type": ["float", "null"]},
            {"name": "sigmapsf_corr", "type": ["float", "null"]},
            {"name": "sigmapsf_corr_ext", "type": ["float", "null"]},
            {"name": "corrected", "type": "boolean"},
            {"name": "dubious", "type": "boolean"},
            {"name": "parent_candid", "type": ["long", "null"]},
            {"name": "has_stamp", "type": "boolean"},
            {"name": "step_id_corr", "type": "string"},
        ],
    },
}
```

#### NonDetections
```json
{
    "type": "array",
    "items": {
        "name": "non_detections_record",
        "type": "record",
        "fields": [
            {"name": "oid", "type": "string"},
            {"name": "mjd", "type": "float"},
            {"name": "diffmaglim", "type": "float"},
            {"name": "fid", "type": "int"},
        ],
    },
}
```

#### Xmatch
```json
{
    "type": "map",
    "values": {"type": "map", "values": ["string", "float", "null"]},
}
```

#### Metadata
```json
{
    "name": "metadata",
    "type": "record",
    "fields": [
        {
            "name": "ps1",
            "type": {
                "name": "ps1",
                "type": "record",
                "fields": [
                    {"name": "simag1", "type": ["float", "null"]},
                    {"name": "objectidps3", "type": ["double", "null"]},
                    {"name": "objectidps1", "type": ["double", "null"]},
                    {"name": "unique1", "type": ["boolean", "null"]},
                    {"name": "unique2", "type": ["boolean", "null"]},
                    {"name": "szmag2", "type": ["double", "null"]},
                    {"name": "srmag3", "type": ["float", "null"]},
                    {"name": "sgscore1", "type": ["float", "null"]},
                    {"name": "szmag3", "type": ["float", "null"]},
                    {"name": "srmag1", "type": ["float", "null"]},
                    {"name": "sgmag1", "type": ["float", "null"]},
                    {"name": "szmag1", "type": ["float", "null"]},
                    {"name": "distpsnr1", "type": ["float", "null"]},
                    {"name": "sgscore2", "type": ["float", "null"]},
                    {"name": "candid", "type": ["long", "null"]},
                    {"name": "simag3", "type": ["float", "null"]},
                    {"name": "objectidps2", "type": ["double", "null"]},
                    {"name": "srmag2", "type": ["float", "null"]},
                    {"name": "unique3", "type": ["boolean", "null"]},
                    {"name": "sgmag3", "type": ["float", "null"]},
                    {"name": "sgmag2", "type": ["double", "null"]},
                    {"name": "simag2", "type": ["float", "null"]},
                    {"name": "distpsnr2", "type": ["float", "null"]},
                    {"name": "distpsnr3", "type": ["float", "null"]},
                    {"name": "nmtchps", "type": ["int", "null"]},
                    {"name": "sgscore3", "type": ["float", "null"]},
                ],
            },
        },
        {
            "name": "ss",
            "type": {
                "name": "ss",
                "type": "record",
                "fields": [
                    {"name": "ssdistnr", "type": ["double", "null"]},
                    {"name": "ssmagnr", "type": ["double", "null"]},
                    {"name": "ssnamenr", "type": ["string", "null"]},
                ],
            },
        },
        {
            "name": "reference",
            "type": {
                "name": "reference",
                "type": "record",
                "fields": [
                    {"name": "magnr", "type": "float"},
                    {"name": "ranr", "type": "float"},
                    {"name": "field", "type": "int"},
                    {"name": "chinr", "type": "float"},
                    {"name": "mjdstartref", "type": "float"},
                    {"name": "mjdendref", "type": "float"},
                    {"name": "decnr", "type": "float"},
                    {"name": "sharpnr", "type": "float"},
                    {"name": "candid", "type": "long"},
                    {"name": "nframesref", "type": "int"},
                    {"name": "rcid", "type": "int"},
                    {"name": "rfid", "type": "int"},
                    {"name": "fid", "type": "int"},
                    {"name": "sigmagnr", "type": "float"},
                ],
            },
        },
        {
            "name": "gaia",
            "type": {
                "name": "gaia",
                "type": "record",
                "fields": [
                    {"name": "maggaiabright", "type": ["float", "null"]},
                    {"name": "neargaiabright", "type": ["float", "null"]},
                    {"name": "unique1", "type": "boolean"},
                    {"name": "neargaia", "type": ["float", "null"]},
                    {"name": "maggaia", "type": ["float", "null"]},
                ],
            },
        },
    ],
}
```


## Build docker image
For use this step, first you must build the image of docker. After that you can run the step for use it.

```bash
docker build -t features_step:version .
```

## Run step

### Run container of docker
You can use a `docker run` command, you must set all environment variables.
```bash
docker run --name my_features_step -e DB_NAME=myhost -e [... all env ...] -d features_step:version
```

### Run docker-compose
Also you can edit the environment variables in [`docker-compose.yml`](https://github.com/alercebroker/feature_step/blob/main/docker-compose.yml) file. After that use `docker-compose up` command. This run only one container.

```bash
docker-compose up -d
```

If you want scale this container, you must set a number of containers to run.

```bash
docker-compose up -d --scale features=32
```

**Note:** Use `docker-compose down` for stop all containers.

### Run the released image

For each release an image is uploaded to ghcr.io that you can use instead of building your own. To do that replace `docker-compose.yml` or the `docker run` command with this image:

```bash
docker pull ghcr.io/alercebroker/feature_step:latest
```

## Using Poetry to manage dependencies

Poetry is configured to manage all dependencies in three groups: main, dev and test. 

### Set-up poetry:
- Install poetry: `pip install poetry`
- If you want to set create `.venv` environment in the project folder: `poetry config virtualenvs.in-project true`
- Create environment with all dependencies (main, dev and test): `poetry install`
- To install only main dependencies: `poetry install --only main`
- Show tree of dependencies: `poetry show --tree`
- Add a new dependency 
  - `poetry add PACKAGE`
  - `poetry add -G dev PACKAGE`
  - `poetry add -G test PACKAGE`

### Use poetry
To use Poetry, you can either 
- `poetry run COMMAND`
OR activate poetry environament and run commands from there
- `poetry shell` or `source PATH/.venv/bin/activate`
