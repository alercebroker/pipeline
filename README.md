Alert Processing Framework (*apf*)
================================================

[![Documentation Status](https://readthedocs.org/projects/apf/badge/?version=latest)](https://apf.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/alercebroker/APF/branch/master/graph/badge.svg)](https://codecov.io/gh/alercebroker/APF)


*apf* is a framework developed to create a dockerized pipeline to
process an alert stream, that can be easily be deployed in a local
machine or distributed using [Kubernetes](https://kubernetes.io).

First developed to process [ZTF data](https://www.ztf.caltech.edu/) it
is capable to be used for any stream/static data processing pipeline.

Installing *apf*
================

*apf* installation can be done with *pip*

``` {.sourceCode .bash}
pip install apf_base
```

This will install the *apf* python package and *apf* command line
script.

*apf* design
============

*apf* is based on *steps* conected through [Apache
Kafka](https://kafka.apache.org/) topics.

Each *step* is composed by a `consumer` and is isolated from other
steps inside a docker container.

When running, the step calls the **execute()** method for each *message* or *message batch*
consumed. A step can have multiple producers and databases back-ends
plugins that can be accessed inside the *execute* method to have a more
complex logic.


<p align="center">
  <img src="docs/source/_static/images/apf-flow.png">
</p>


This generic step greatly reduce the development of each component of
the pipeline and make it easier to test each component separately.


*apf* features
==============

- Automatic Metric Sender (KafkaMetrics)
- Automatic Code Generation (`apf new-step <step_name>`)
- Multiple Consumer Plugins:
  - Kafka
  - AVRO
  - CSV
  - JSON
- Producers:
  - Kafka
  - CSV
- Metrics:
  - Kafka

*apf* Quickstart
===================

A **quick-start** guide to create a new step can be found [here](https://apf.readthedocs.io/en/latest/new_step.html).
