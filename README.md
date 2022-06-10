# Alerce Reflector Step

Copies one or more Kafka topics into a new one. Used to replicate messages from
an external source into the current Kafka cluster.

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
