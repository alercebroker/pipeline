# Xmatch Step

## Description

This step consumes directly from the originally ZTF stream, performs a crossmatch with ALLWISE catalog and sends the result to the xmatch topic of the Kafka server. The output topic will be eventually consumed by the Stamp Classifier and Late Classification steps at leat.

#### Previous steps: 
- None

#### Next steps:
- [Stamp Classifier](https://github.com/alercebroker/stamp_classifier_step)
- [Late Classification](https://github.com/alercebroker/late_classification_step)

## Previous conditions

No special conditions, only connection to kafka.

## Version
- **0.0.1:** 
	- Use of APF
	- Use of CDS Xmatch Client
  - Crossmatch only with ALLWISE catalog
	- Future enhancement: 
		- Crossmatch with more catalogs provided by CDS: GAIA-DR2, SDSS-DR12 
		- Define whether or not save results to a database and how

## Libraries used
- APF
- Pandas
- Astropy
- [CDS Xmatch Client](https://github.com/alercebroker/cds_xmatch_client)

## Environment variables

### Consumer setup

- `CONSUMER_TOPICS`: Input topic as a single element array
- `CONSUMER_SERVER`: Kafka host with port. e.g: `localhost:9092`
- `CONSUMER_GROUP_ID`: Name for consumer group. e.g: `xmatch`

### Producer setup

- `PRODUCER_TOPIC`: Name of output topic. e.g: `xmatch`
- `PRODUCER_SERVER`: Kafka host with port. e.g: `localhost:9092`

### Xmatch setup

- `CATALOG`: An array of catalog settings to perform crossmatch with. Each catalog contains:
	- `name` : Name of the catalog
	- `columns` : A subset of columns to be selected from the catalog

## Stream

This step require a consumer and producer.

### Input schema

[Documentation of ZTF Avro schema.](https://zwickytransientfacility.github.io/ztf-avro-alert/schema.html) 
[Avsc files](https://github.com/ZwickyTransientFacility/ztf-avro-alert/tree/master/schema)


### Output schema
```Python

RESULT_TYPE = {
	'type' : 'map',
	'values' : {
		'type' : 'map',
		'values' : [ "string", "float", "null"]
	}
}

{
	'doc' : 'Xmatch',
	'name' : 'xmatch',
	'type' : 'record',
	'fields' : [
			{'name' : 'oid', 'type' : 'string'},
			{'name' : 'result', 'type' : [ "null", RESULT_TYPE ] }
	]
}



```
