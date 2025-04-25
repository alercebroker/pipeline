from confluent_kafka import Consumer, KafkaException
import fastavro
import json
import io
import yaml

with open('tests/consumer/kafka_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

conf = config['kafka_config1']
conf2 = config['kafka_config2']


def consume_and_save_json(consumer_config, topic, schema_path, output_file):
    consumer = Consumer(consumer_config)
    consumer.subscribe([topic])

    try:
        while True:
            msg = consumer.poll(1.0)
            print(msg)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            else:
                schema = fastavro.schema.load_schema(schema_path)
                bytes_io = io.BytesIO(msg.value())
                data = fastavro.schemaless_reader(bytes_io, schema)
                print(data)
                return data  
    finally:
        consumer.close()

data1 = consume_and_save_json(
    conf, 
    'correction', 
    '/home/kay/Descargas/pipeline_2204/pipeline/schemas/correction_step/output.avsc',
    'correction_data.json'
)

data2 = consume_and_save_json(
    conf2, 
    'correction-ms-ztf', 
   '/home/kay/Descargas/pipeline_2204/pipeline/schemas/correction_ms_ztf/output.avsc',
    'correction_ms_ztf_data.json'
)

print("Both topics processed successfully.")