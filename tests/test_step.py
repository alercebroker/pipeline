import unittest
import os
import sys
import subprocess
import docker
import time
from click.testing import CliRunner
from correction.step import Correction
from apf.consumers import AVROFileConsumer
from db_plugins.cli.manage import init_sql
from settings import DB_CONFIG

FILE_PATH = os.path.dirname(__file__)
STEP_PATH = os.path.join(FILE_PATH, "..")

sys.path.append(STEP_PATH)

class StepTest(unittest.TestCase):
    container_name = "test_postgres"
    container = None

    def setUp(self):
        self.client = docker.from_env()
        self.container = self.client.containers.run(image="postgres", name=self.container_name,
                              environment=["POSTGRES_USER=postgres", "POSTGRES_PASSWORD=password", "POSTGRES_DB=test"],
                              detach=True, ports={'5432/tcp': 5432} )


        time.sleep(5)
        subprocess.run([f'dbp initdb --settings_path {os.path.join(FILE_PATH, "settings.py")}'], shell=True)

    def test_execute(self):
        CONSUMER_CONFIG = {
            "DIRECTORY_PATH": os.path.join(FILE_PATH,"examples/avro_test")
        }
        PRODUCER_CONFIG = {
            "CLASS": 'apf.producers.GenericProducer'
        }
        step = Correction(consumer=AVROFileConsumer(CONSUMER_CONFIG),
                          config={
                                "DB_CONFIG": DB_CONFIG,
                                "PRODUCER_CONFIG": PRODUCER_CONFIG,
                                "STEP_VERSION": os.getenv("STEP_VERSION", "dev")
                                }
                            )
        step.start()

    def tearDown(self):
        self.container.remove(force=True)
