import os
import sys
from typing import Any, Iterable
from random import Random


from apf.core import get_class
from apf.core.settings import config_from_yaml_file
from apf.core.step import GenericProducer
from tqdm import tqdm

sys.path.append("tests")

from generator.lsst_alert import LsstAlertGenerator

from data.generator_ztf import (  # pyright: ignore
    generate_alerts as generate_alerts_ztf,
)

PRODUCE_SAMPLE_CONFIG: dict[str, Any] = config_from_yaml_file(
    os.getenv("CONFIG_PRODUCE_SAMPLE_YAML_PATH", "config.produce_sample.yaml")
)

rng = Random(42)
generator_lsst = LsstAlertGenerator(rng=rng, new_obj_rate=0.1)
GENERATORS = {"lsst": generator_lsst, "ztf": generate_alerts_ztf}

class SampleProducer:
    producer: GenericProducer

    def __init__(self, config: dict[str, Any]):
        Producer = get_class(config["CLASS"])

        self.producer = Producer(config)
        self.producer.set_key_field("alertId")

    def produce(self, msgs: Iterable[dict[str, Any]], flush: bool = True):
        for msg in tqdm(msgs):
            self.producer.produce(msg, flush=False)
        if flush:
            self.flush()

    def flush(self):
        self.producer.producer.flush()

def produce():
    survey = PRODUCE_SAMPLE_CONFIG.get("SURVEY", "lsst")
    assert type(survey) is str
    survey = survey.lower()
    assert survey in GENERATORS.keys()

    n_messages = PRODUCE_SAMPLE_CONFIG.get("N_MESSAGES", 100)
    assert type(n_messages) is int
    assert n_messages > 0

    survey = survey
    generator = GENERATORS[survey]
    producer = SampleProducer(PRODUCE_SAMPLE_CONFIG["PRODUCER_CONFIG"])
    alerts = [generator.generate_alert() for _ in range(n_messages)]

    batch_size = PRODUCE_SAMPLE_CONFIG.get("BATCH_SIZE", 100)
    for i in range(0, len(alerts), batch_size):
        batch = alerts[i:i + batch_size]
        producer.produce(batch, flush=False)
    producer.flush() 
