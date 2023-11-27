# Test utils and mock data generation

This library is a tool to provide common services for testing steps.

## Example usage:

``` python
## step conftest.py
from test_utils.fixtures import *

def docker_compose_file():
    return "my_path/to_docker_compose/docker-compose.yaml"
```

```python
## test_file.py
def example_populate_mongo(mongo_service: str):
    # do something with mongo

def example_populate_psql(psql_service: str):
    # do something with psql

def example_populate_kafka(kafka_service: str):
    # do something with kafka

def test_something(kafka_service, mongo_service, psql_service):
    example_populate_mongo(mongo_service)
    example_populate_psql(psql_service)
    example_populate_kafka(kafka_service)
    # do the thing connecting to services
    # ...
    # do assertions
```

``` yaml
version: "3"
services:
  zookeeper:
    image: 'bitnami/zookeeper:latest'
    ports:
      - '2181:2181'
    environment:
      - ALLOW_ANONYMOUS_LOGIN=yes

  kafka:
    image: 'bitnami/kafka:3.3.1'
    ports:
      - '9092:9092'
    environment:
      - KAFKA_BROKER_ID=1
      - KAFKA_CFG_LISTENERS=PLAINTEXT://:9092
      - KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://127.0.0.1:9092
      - KAFKA_CFG_ZOOKEEPER_CONNECT=zookeeper:2181
      - ALLOW_PLAINTEXT_LISTENER=yes
    depends_on:
      - zookeeper

  mongo:
    image: mongo
    ports:
      - "27017:27017"
    command: mongod --notablescan

  postgres:
    image: postgres
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_DB=postgres
      - POSTGRES_PASSWORD=postgres
    ports:
      - "5432:5432"
```
