# LC Correction (preprocess) step

Last update: 2020/10/02

## Description

Light curve correction from detections of ZTF data stream.

- `isdiffpos`: Apply sign in raw data. `isdiffpos` from stream is `"t"`, `"f"`, `"1"` or `"-1"`. We cast the value to `1` or `-1`.
- Correct magnitude: Calculate this correction for `magpsf` and `magap`.
- Correct sigma magnitude: Calculate this correction for `sigmagpsf` and `sigmagap`.
- Process and check previous candidates.

Get magnitude statistics from light curve corrected (each band).

- Max, min, mean, median of magnitude and sigma (both corrected).
- First and last detection.
- Number of detections.
- Number of dubious correction.
- Get dm/dt.


Get object statistics:

- Number of total detections.
- Discovery date.
- Last date of observation to object.
- Mean of RA and Dec.
- If object was corrected.
- And more.


Get metadata respect to:

- Gaia information from ZTF.
- Solar system information from ZTF.
- Pan-STARRS information form ZTF.
- Reference information from ZTF.

#### Previous steps:

- None

#### Next steps:

- [LC Features step](https://github.com/alercebroker/feature_step)

## Database interactions


### Select:

- Query for get the light curve (detections and non detections).
- Query if object exists in database.

### Insert:

- New detection.
- New non-detection(s).
- Metadata: Solar System (ss_ztf), Panstarr (ps1_ztf), Gaia (gaia_ztf), Reference (reference), Magnitude Statistics (magstats)

## Previous conditions

No special conditions, only connection to kafka, database and elasticsearch.

## Version

- **0.0.2:**
	- Added unittests.
	- Changed KafkaProducer as hardcoded producer to configurable through settings.py.

- **0.0.1:**
	- Use of APF. - Implementation of functions to correct magnitudes.
	- Possible enhance: - [X] Use of `astropy` to convert `jd` to `mjd`. - [X] Optimization in [for-loop](https://github.com/alercebroker/correction_step/blob/master/correction/step.py#L146).

## Libraries used

- Astropy
- Numpy
- [APF](https://github.com/alercebroker/APF)
- [LC Correction](https://github.com/alercebroker/lc_correction)
- [DB Plugins](https://github.com/alercebroker/db-plugins)

## Configure Step Producer

In settings.py:

```python
PRODUCER_CONFIG = {
	"CLASS": '<Class import path>',
	...
}
```

Use an `apf.producer.GenericProducer` type class, i.e. `apf.producer.KafkaProducer`

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

### Metrics setup

- `METRICS_HOST`: Kafka brokers to send the metrics.
- `METRICS_TOPIC`: Topic to save the metrics.

## Stream

This step require a consumer and producer.

### Input schema

[Documentation of ZTF Avro schema.](https://zwickytransientfacility.github.io/ztf-avro-alert/schema.html)
[Avsc files](https://github.com/ZwickyTransientFacility/ztf-avro-alert/tree/master/schema)

### Output schema

```Python
{
  "doc": "Lightcurve",
  "name": "lightcurve",
  "type": "record",
  "fields": [
    {
      "name": "oid",
      "type": "string"
    },
    {
      "name": "candid",
      "type": "string"
    },
    {
      "name": "fid",
      "type": "int"
    }
    {
      "name": "detections",
      "type": {
        "type": "array",
        "items": {
          "type": "record",
          "name": "detection",
          "fields": [
            {
              "name": "candid",
              "type": "string"
            },
            {
              "name": "mjd",
              "type": "float"
            },
            {
              "name": "fid",
              "type": "int"
            },
            {
              "name": "magpsf_corr",
              "type": [
                "float",
                "null"
              ],
              "default": None
            },
            {
              "name": "magap_corr",
              "type": [
                "float",
                "null"
              ],
              "default": None
            },
            {
              "name": "sigmapsf_corr",
              "type": [
                "float",
                "null"
              ],
              "default": None
            },
            {
              "name": "sigmagap_corr",
              "type": [
                "float",
                "null"
              ],
              "default": None
            },
            {
              "name": "ra",
              "type": "float"
            },
            {
              "name": "dec",
              "type": "float"
            },
            {
              "name": "rb",
              "type": [
                "float",
                "null"
              ],
              "default": None
            },
            {
              "name": "oid",
              "type": "string"
            },
            {
              "name": "alert",
              "type": {
                "type": "map",
                "values": [
                  "int",
                  "float",
                  "string",
                  "null"
                ]
              }
            }
          ]
        }
      }
    },
    {
      "name": "non_detections",
      "type": {
        "type": "array",
        "items": {
          "type": "map",
          "values": [
            "float",
            "int",
            "string",
            "null"
          ]
        }
      }
    }
  ]
}
```

## Build docker image

For use this step, first you must build the image of docker. After that you can run the step for use it.

```bash
docker build -t correction_step:version .
```

## Run step

### Run container of docker

You can use a `docker run` command, you must set all environment variables.

```bash
docker run --name my_correction_step -e DB_HOST=myhost -e [... all env ...] -d correction_step:0.0.1
```

### Run docker-compose

Also you can edit the environment variables in [`docker-compose.yml`](https://github.com/alercebroker/correction_step/blob/master/docker-compose.yml) file. After that use `docker-compose up` command. This run only one container.

```bash
docker-compose up -d
```

If you want scale this container, you must set a number of containers to run.

```bash
docker-compose up -d --scale correction_step=n
```

**Note:** Use `docker-compose down` for stop all containers.
