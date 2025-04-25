from confluent_kafka import Consumer, KafkaException
import fastavro
import json
import io
import yaml

with open('/tests/consumer/kafka_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

list_oid_string = ["ZTF19aadfgff", "ZTF19aaeskae", "ZTF20adcbuvi"]
list_oid_int = [36028949624820311, 36028949625508348, 36028957691757218]

conf = config['kafka_config1']
conf2 = config['kafka_config2']

def consume_and_save_json(consumer_config, topic, schema_path, output_file, list_oids):
    consumer = Consumer(consumer_config)
    consumer.subscribe([topic])
    
    try:
        while True:
            if not list_oids:
                print("All data processed successfully (or i hope so! team grafini was here).") #clea :v fokiu xDljdkslajdkalsjdaklasdjfklasdjfl pero dde se guarda la datta, por fuera? yeyeye, crea un jason (ME EQUIVOQUE EN ALGO ) yoooo
                break
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            else:
                schema = fastavro.schema.load_schema(schema_path)
                bytes_io = io.BytesIO(msg.value())
                data = fastavro.schemaless_reader(bytes_io, schema)
                oid = data["oid"]
                print(oid)
                if oid in list_oids:
                    list_oids.remove(oid)
                    # Save to JSON file
                    oid_str = str(oid)
                    output_path = oid_str + "_" + output_file
                    with open(output_path, 'w') as json_file:
                        json.dump(data, json_file, indent=2)
                    print(f"Data saved to {output_file}")
                    
                
    finally:
        consumer.close()


data1 = consume_and_save_json(
    conf, 
    'correction', 
    '/home/kay/Descargas/pipeline_2204/pipeline/schemas/correction_step/output.avsc',
    'correction_data.json',
    list_oid_string
)

## Process second topic
data2 = consume_and_save_json(
    conf2, 
    'correction-ms-ztf', 
    '/home/kay/Descargas/pipeline_2204/pipeline/schemas/correction_ms_ztf/output.avsc',
    'correction_ms_ztf_data.json',
    list_oid_int
)

print("Both topics processed successfully.")