# The streaming step pulls heavy deps (confluent_kafka, apf, prometheus_client,
# xmatch_client) that the offline tooling doesn't need. Tolerate those being
# absent so `features.offline.*` imports in a lean env, but never hide a genuine
# import bug inside step.py.
try:
    from . import step
    from .step import *
except ImportError as e:
    if e.name not in (
        "confluent_kafka",
        "apf",
        "prometheus_client",
        "xmatch_client",
    ):
        raise
