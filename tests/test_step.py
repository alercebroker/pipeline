import unittest
import subprocess
import time
import sys
import os
import docker
from settings import DB_CONFIG
from apf.consumers import AVROFileConsumer
FILE_PATH = os.path.dirname(__file__)
STEP_PATH = os.path.join(FILE_PATH,"..")
sys.path.append(STEP_PATH)
from correction import Correction

class StepTest(unittest.TestCase):
    container_name = "test_postgres"
    container = None

    def setUp(self):
        self.client = docker.from_env()
        self.container = self.client.containers.run(
                image="postgres", name=self.container_name,
                environment=["POSTGRES_USER=postgres", "POSTGRES_PASSWORD=password", "POSTGRES_DB=test"],
                detach=True, ports={'5432/tcp':5432}
        )

        time.sleep(5)
        subprocess.run([f'dbp initdb --settings_path {os.path.join(FILE_PATH, "settings.py")}'], shell=True)

    def test_execute(self):
        CONSUMER_CONFIG={
            "DIRECTORY_PATH": os.path.join(FILE_PATH, "examples/avro_test")
        }
        PRODUCER_CONFIG={
            "CLASS": 'apf.producers.GenericProducer'
        }

        STEP_METADATA = {
            "STEP_VERSION": "dev",
            "STEP_ID": "preprocess",
            "STEP_NAME": "preprocess",
            "STEP_COMMENTS": "",
        }
        step = Correction(consumer=AVROFileConsumer(CONSUMER_CONFIG),
                          config={
                            "DB_CONFIG": DB_CONFIG,
                            "PRODUCER_CONFIG":PRODUCER_CONFIG,
                            "STEP_METADATA": STEP_METADATA
                          }
                )
        step.start()

    def tearDown(self):
        time.sleep(5)
        self.container.stop()
        self.container.remove()

if __name__=="__main__":
    t = StepTest()
    t.test_execute()
