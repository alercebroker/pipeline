# S3 Upload

## Description

- Upload avro files to bucket in AWS.

#### Previous steps: 
- None

#### Next steps:
- None

## Database interactions
No interaction.

## Previous conditions

Credentials of AWS.

## Version
- **0.0.1:** 
	- Use of APF.
	- Use of boto3.

## Libraries used
- APF
- boto3

## Environment variables

### Consumer setup

- `CONSUMER_TOPICS`: Some topics. String separated by commas. e.g: `topic_one` or `topic_two,topic_three`
- `CONSUMER_SERVER`: Kafka host with port. e.g: `localhost:9092`
- `CONSUMER_GROUP_ID`: Name for consumer group. e.g: `correction`

### Elasticsearch setup
- `ES_PREFIX`: Enables the indexing of term prefixes to speed up prefix searches. e.g: `ztf_pipeline`
- `ES_NETWORK_HOST`: Elasticsearch host.
- `ES_NETWORK_PORT`: Elasticsearch port.

### S3 setup
- `BUCKET_NAME`: Name of bucket to store avro files.
- `AWS_ACCESS_KEY_ID`: Access key id of your AWS account.
- `AWS_SECRET_ACCESS_KEY`: Secret access key of your AWS account.

## Stream

This step require only consumer.

### Input schema
- [Documentation of ZTF Avro schema.](https://zwickytransientfacility.github.io/ztf-avro-alert/schema.html)
- [Avsc files](https://github.com/ZwickyTransientFacility/ztf-avro-alert/tree/master/schema)
### Output schema
- None

## Build docker image
For use this step, first you must build the image of docker. After that you can run the step for use it.

```bash
docker build -t s3_step:0.0.1 . 
```

## Run step

### Run container of docker
You can use a `docker run` command, you must set all environment variables.
```bash
docker run --name my_s3_step -e BUCKET_NAME=myhost -e [... all env ...] -d s3_step:0.0.1
```

### Run docker-compose
Also you can edit the environment variables in [`docker-compose.yml`](https://github.com/alercebroker/s3_step/blob/master/docker-compose.yml) file. After that use `docker-compose up` command. This run only one container.

```bash
docker-compose up -d
```

If you want scale this container, you must set a number of containers to run.

```bash
docker-compose up -d --scale s3_step=32
```

**Note:** Use `docker-compose down` for stop all containers.
