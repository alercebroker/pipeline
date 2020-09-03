import unittest
import subprocess
import time
import sys
import os
import docker
from settings import DB_CONFIG
from apf.consumers import JSONConsumer
import json
FILE_PATH = os.path.dirname(__file__)
STEP_PATH = os.path.join(FILE_PATH,"..")
sys.path.append(STEP_PATH)
from late_classification import LateClassifier


from db_plugins.db.sql import SQLConnection
from db_plugins.db.sql.models import (
                    Object,
                    )

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
        resp = os.system(f'dbp initdb --settings_path {os.path.join(FILE_PATH, "settings.py")}')
        with open(os.path.join(FILE_PATH, "examples/objects_data.json")) as f:
            objects = json.load(f)
        self.driver = SQLConnection()
        self.driver.connect(DB_CONFIG["SQL"])
        self.driver.session.query().bulk_insert(objects, Object)
        self.driver.session.commit()



    def test_execute(self):
        import logging
        level=logging.DEBUG

        logging.basicConfig(level=level,
                            format='%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',)


        CONSUMER_CONFIG={
            "FILE_PATH": os.path.join(FILE_PATH, "examples/test_features_small.json")
        }
        PRODUCER_CONFIG={
            "CLASS": 'apf.producers.GenericProducer'
        }
        step = LateClassifier(consumer=JSONConsumer(CONSUMER_CONFIG),
                          config={
                            "DB_CONFIG": DB_CONFIG,
                            "PRODUCER_CONFIG":PRODUCER_CONFIG,
                            "STEP_VERSION": "test"
                          }
                )
        step.start()

    # def tearDownClass(self):
    #    self.container.stop()
    #    self.container.remove()

if __name__=="__main__":
    t = StepTest()
    t.test_execute()
