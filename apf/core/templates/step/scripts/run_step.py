import os
import sys

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH,".."))

sys.path.append(PACKAGE_PATH)

from {{package_name}} import {{class_name}}

from apf.consumers import GenericConsumer
from apf.producers import GenericProducer


consumer = GenericConsumer()
producer = GenericProducer()

step = {{class_name}}(consumer,producer)
step.execute()
