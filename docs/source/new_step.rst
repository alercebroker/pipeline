Creating a new apf step
=======================

*apf* was created to simplify the development of an stream processing pipeline.

To illustrate how the creation of a pipeline step was intended we have the following diagram.

.. image:: _static/images/apf-flow.png
    :align: center

This tutorial will guide developers to create an example step from the installation of the framework until building and running the docker image locally.


1. Installing *apf*
----------------------

To install *apf* run

.. code-block:: bash

    pip install apf_base

This will install the package and a command line script.

.. code-block:: bash

    apf [--help] command

2. Creating base step
----------------------

*apf* comes with a code generation tool to create a base for a new step.

To create this base run

.. code-block:: bash

    apf new-step example_step

This command will create the following file tree

.. code-block:: text

    example_test/
    ├── example_test/
    │   ├── __init__.py
    │   └── step.py
    ├── scripts/
    │   ├── run_multiprocess.py
    │   └── run_step.py
    ├── tests/
    ├── Dockerfile
    ├── requirements.txt
    └── settings.py

The step will be a python package called `example_test`, inside the package there is
a `step.py` with the step logic.

3. Coding the step
----------------------

In `example_test/step.py` we will code the step logic, it can be as simple as printing
the message or a more complex logic. For each new message the :func:`execute()` method is called with
a python :class:`dict` with the message itself.

.. code-block:: python

    #example_test/step.py
    def execute(self,message):
      ################################
      #   Here comes the Step Logic  #
      ################################

      pass

For this example we will just log the message changing the execution code to

.. code-block :: python

    #example_test/step.py
    def execute(self,message):
      # Logging the message
      self.logger.info(message)


Here :attr:`self.logger` is the default logger (`logging.Logger`) from :class:`apf.core.GenericStep`.

Then we can go to `scripts/run_step.py` or `scripts/run_multiprocess.py`
this scripts runs the step, here we can define the consumers, producers and other plugins used in the *step*.

The basic `run_step.py` comes with the following

.. code-block:: python

    #scripts/run_step.py
    if "CLASS" in CONSUMER_CONFIG:
        Consumer = get_class(CONSUMER_CONFIG["CLASS"])
    else:
        from apf.consumers import KafkaConsumer as Consumer

    consumer = Consumer(config=CONSUMER_CONFIG)
    step = ExampleTest(consumer,config=STEP_CONFIG,level=level)
    step.start()


The :class:`apf.consumers.KafkaConsumer` can be changed to another consumer, for example a :class:`apf.consumers.CSVConsumer`
to read a *CSV* file or :class:`apf.consumers.JSONConsumer` to process a JSON file, the default
consumer can be overridden in the `settings.py` file.

.. code-block:: python

    #settings.py
    CONSUMER_CONFIG = {
      'CLASS': 'apf.consumers.KafkaConsumer'
    }

4. Configuring the step
------------------------

After coding the step and modifying the script, the step must be configured.

There are 2 files needed to configure a step.

1- `settings.py`:

  This file contains all the configuration passed to the consumers, producers and plugins. Having it separately from
  the main script make it easier to change configurations from run to run.

  For *good practice* having environmental variables as parameters is better than hard-coding them to the settings file,
  and comes very handy when deploying the same dockerized step with different configurations.

  The basic `settings.py` comes with the following

  .. code-block:: python

    #settings.py
    CONSUMER_CONFIG = {}  #Consumer configuration
    STEP_CONFIG = {
      "N_PROCESS" # Number of prcesses on multi-process script.
    }                     #Step Configuration

  We will test our step with a CSVConsumer

  .. code-block:: python

    #settings.py
    CONSUMER_CONFIG = {
      "CLASS": "apf.consumers.CSVConsumer",
      "FILE_PATH": "https://assets.alerce.online/tutorials/alerce-workshop-sep/pandas-sql/detections.csv",
      "OTHER_ARGS": {
          "index_col": "oid"
      }
    }

2- `requirements.txt`

  The default requirements file for any python package, for *good practice* having the package with and specific version
  is better than using the latest one.

  In this example we are using only the :class:`GenericConsumer()`, there is no need to specify parameters for this consumer.

  The basic `requirements.txt` comes with the current `apf` version as a required package

  .. code-block:: python

    #requirements.txt
    apf==<version>

  By default the *apf* package is already on the requirements file, so for this tutorial we will skip this step.



5. Running the step locally
----------------------------

The step can me executed as a single process with

.. code-block :: bash

  python scripts/run_script.py


Or with `multiprocessing` using

.. code-block :: bash

  python scripts/run_multiprocess.py

The number of process can be configured in `settings.py`, adding `N_PROCESS` to `STEP_CONFIG` variable.

To run the step dockerized, first we need to build the step

.. code-block :: bash

  docker build -t example_step .
  docker run --rm --name example_step example_step


Assuming that *apf* is already installed we can test our new step with

.. code-block:: bash

  python scripts/run_step.py

This will show each row from the CSV file.


.. note::
   Try using another `Consumer` configure it and run it locally to check it works. For example a `CSVConsumer` or a `JSONConsumer`
