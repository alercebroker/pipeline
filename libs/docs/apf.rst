Alert Processing Framework (*apf*) documentation
=============================================================

*apf* is a framework developed to create a dockerized pipeline to process an alert stream, that can be easily be deployed in a local machine or distributed using `Kubernetes <https://kubernetes.io>`_.

First developed to process `ZTF data <https://www.ztf.caltech.edu/>`_ it is capable to be used
for any stream/static data processing pipeline.

Installing *apf*
====================

*apf* installation can be done with *pip*

.. code-block:: bash

      pip install apf_base

This will install the *apf* python package and *apf* command line script.


*apf* design
==============

*apf* is based on *steps* conected through `Apache Kafka <https://kafka.apache.org/>`_ topics.

Each *step* is composed by a **consumer** and is isolated from other steps inside a docker container.

When running, the step calls the **execute()** method for each *message* consumed. A step can have multiple producers and databases back-ends plugins that can be accessed inside the *execute* method to have a more complex
logic.

.. image:: _static/images/step.png
    :align: center

This generic step greatly reduce the development of each component of the pipeline and make it easier to test each component separately.

A quick-start guide to create a new step can be found `here <new_step.html>`_.


Documentation
=============

.. toctree::
  apf.new_step
  apf.step
  apf.basic_metrics
  apf.profiling
  apf.core
  apf.consumers
  apf.producers
  apf.metrics
  :maxdepth: 2

