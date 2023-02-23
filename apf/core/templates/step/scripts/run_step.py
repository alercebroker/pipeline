import os
import sys

import logging

from prometheus_client import start_http_server
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH,".."))

sys.path.append(PACKAGE_PATH)
from settings import *

level = logging.INFO
if 'LOGGING_DEBUG' in locals():
    if LOGGING_DEBUG:
        level=logging.DEBUG

logging.basicConfig(level=level,
                    format='%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',)


from {{package_name}} import {{class_name}}

if PROMETHEUS:
    start_http_server(8000)

step = {{class_name}}(config=STEP_CONFIG,level=level)
step.start()
