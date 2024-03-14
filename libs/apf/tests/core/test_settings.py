from apf.core.settings import config_from_yaml_string


def test_load_from_yaml():
    config = config_from_yaml_string(
        """
    CONSUMER_CONFIG:
      TOPICS: [ztf,atlas]
      CLASS: apf.consumers.KafkaConsumer
      PARAMS:
        bootstrap.servers: localhost:9092
        group.id: test
        enable.partition.eof: true
      consume.timeout: 10
      consume.messages: 1000
    METRICS_CONFIG:
      EXTRA_METRICS:
        - key: "aid"
        - key: "candid"
    """
    )
    assert config["CONSUMER_CONFIG"]["TOPICS"] == ["ztf", "atlas"]
    assert config["CONSUMER_CONFIG"]["CLASS"] == "apf.consumers.KafkaConsumer"
    assert config["CONSUMER_CONFIG"]["PARAMS"]["bootstrap.servers"] == "localhost:9092"
    assert config["CONSUMER_CONFIG"]["PARAMS"]["group.id"] == "test"
    assert config["CONSUMER_CONFIG"]["PARAMS"]["enable.partition.eof"] == True
    assert config["CONSUMER_CONFIG"]["consume.timeout"] == 10
    assert config["CONSUMER_CONFIG"]["consume.messages"] == 1000
    assert config["METRICS_CONFIG"]["EXTRA_METRICS"][0]["key"] == "aid"
    assert config["METRICS_CONFIG"]["EXTRA_METRICS"][1]["key"] == "candid"
