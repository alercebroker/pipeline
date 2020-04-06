# LC Correction

## Description

Light curve correction from detections of ZTF data stream. 

- `isdiffpos`: Apply sign in raw data. `isdiffpos` from stream is `"t"`,  `"f"`, `"1"` or `"-1"`. We cast the value to `1` or `-1`.
- Correct magnitude: Calculate this correction for `magpsf` and `magap`.
- Correct sigma magnitude: Calculate this correction for `sigmagpsf` and `sigmagap`.

#### Previous steps: 
- None

#### Next steps:
- [LC Features step](https://github.com/alercebroker/feature_step)
- [LC Statistics](https://github.com/alercebroker/magnitude_statistics_step)

## Database interactions

This step interact with some tables of `new_pipeline` database. 

### Select:
- The function [`get_ligthcurve`](https://github.com/alercebroker/correction_step/blob/master/correction/step.py#L202) select detections and non-detections of an object. 

### Insert:
- New detection.
- New non-detection(s).

## Previous conditions

No special conditions, only connection to kafka, database and elasticsearch.

## Version
- **0.0.1:** 
	- Use of APF.
	- Implementation of functions to correct magnitudes.
	- Possible enhance: 
		- [X] Use of `astropy` to convert `jd` to `mjd`. 
		- [ ] Optimization in [for-loop](https://github.com/alercebroker/correction_step/blob/master/correction/step.py#L146).

## Libraries used
- APF
- Numpy
- Astropy

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
docker build -t correction_step:0.0.1 . 
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
docker-compose up -d --scale correction_step=32
```

**Note:** Use `docker-compose down` for stop all containers.
