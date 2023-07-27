Metrics tutorial
=======================

*apf* has a simple metrics system to check the step performance.

In this tutorial we will create a simple logfile with some metrics, to create a basic
step first check `this tutorial <new_step.html>`_.

1. Adding a metrics producer
-----------------------------

The simplest metrics producer is :class:`apf.metrics.LogfileMetricsProducer`.

Adding a metrics producer to a *apf* step is done adding it to the `setting.py` file.

.. code-block:: python

    #settings.py
    METRICS_CONFIG = {
        "CLASS": "apf.metrics.LogfileMetricsProducer",
        "PARAMS": {}
    }

    ## Step Configuration
    STEP_CONFIG = {
        "METRICS_CONFIG": METRICS_CONFIG
    }

First we create a new configuration (`METRICS_CONFIG`) with the metrics producer class and the producer parameters. Then we can run the step with

.. code-block:: bash

    python scripts/run_step.py

This will show something like

.. code-block:: bash

    2020-10-14 12:46:16 INFO CSVConsumer.__init__: Creating CSVConsumer
    2020-10-14 12:46:16 INFO TestStep.__init__: Creating TestStep
    2020-10-14 12:46:16 INFO LogfileMetricsProducer.__init__: Creating LogfileMetricsProducer
    2020-10-14 12:46:16 INFO LogfileMetricsProducer.__init__: Writing metrics logs into /tmp/logfilemetricsproducer-ehtv7171.log

If no parameters are given to the :class:`apf.metrics.LogfileMetricsProducer` the file will be created in /tmp.

2. Configuring the producer
----------------------------

Each metrics producer has different parameters, in the case of the :class:`apf.metrics.LogfileMetricsProducer` we only need a PATH parameter.

.. code-block:: python

    #settings.py
    METRICS_CONFIG = {
        "CLASS": "apf.metrics.LogfileMetricsProducer",
        "PARAMS": {
          "PATH": "logs/example.log"
        }
    }

    ## Step Configuration
    STEP_CONFIG = {
        "METRICS_CONFIG": METRICS_CONFIG
    }

This will ensure to create the path and the logfile.

3. Adding more metrics.
-----------------------

By default a *apf* step will send the following metrics:

- timestamp_received: When the message was received.
- timestamp_sent: The moment that the message was processed and sended to other step.
- execution_time: Time between received and sended.
- n_messages: number of messages processed (useful for batch processing).
- source: Step name.

but we can add more data from the message to be sended with the EXTRA_METRICS parameter.

The EXTRA_METRICS parameter uses a list of possible metrics

.. code-block:: python

    #settings.py
    EXTRA_METRICS = ["candid", "ra", "dec"]

For example we add the candid, right ascension and declination as extra metrics in the
metrics configuration.

.. code-block:: python

    #settings.py
    METRICS_CONFIG = {
        "CLASS": "apf.metrics.LogfileMetricsProducer",
        "EXTRA_METRICS": EXTRA_METRICS,
        "PARAMS": {
          "PATH": "logs/example.log"
        }
    }

    ## Step Configuration
    STEP_CONFIG = {
        "METRICS_CONFIG": METRICS_CONFIG
    }

Now we have more metrics to trace our messages. But if we can process the metrics values
before sending it, we also can add dictionaries to EXTRA_METRICS.

For example we want to transform the Modified Julian Date (*mjd*) field into a date, for this we need to create a function
to process the value.

.. code-block:: python

    #settings.py
    from astropy.time import Time
    def mjd_to_date(mjd):
        t = Time(mjd, format="mjd")
        dt = t.datetime
        dt_str = dt.strftime("%m/%d/%Y, %H:%M:%S")
        return dt_str

(Make sure that astropy is installed and added to the requirements.txt file)

Then we add this formatting function to our EXTRA_METRICS as a dictionary.

- The **required** parameter is "key", the step will use this key to get the value from the message.
- If a "format" parameter is passsed the function is called on the raw value.
- If there is a "alias" parameter the metric name will have that name.

(Both format and alias can be used independently)

.. code-block:: python
    #settings.py
    mjd_formatting = {
            "key": "mjd",
            "format": mjd_to_date,
            "alias": "mjd_date"}

    EXTRA_METRICS = ["candid", "ra", "dec", mjd_formatting]

    METRICS_CONFIG = {
        "CLASS": "apf.metrics.LogfileMetricsProducer",
        "EXTRA_METRICS": EXTRA_METRICS,
        "PARAMS": {
          "PATH": "logs/example.log"
        }
    }

    ## Step Configuration
    STEP_CONFIG = {
        "METRICS_CONFIG": METRICS_CONFIG
    }

Now our metrics will have a new field called *mjd_date*, and will call the function for each execution.

An example from *example.log*

.. code-block:: 

  timestamp_received: 2020-10-14 16:59:46.356606+00:00, timestamp_sent: 2020-10-14 16:59:46.356665+00:00, execution_time: 5.9e-05, candid: 1221211305915015000, ra: 145.9124706, dec: -7.018032400000001, date_mjd: 05/06/2020, 05:04:17, n_messages: 1, source: TestStep


4. Sending metrics inside the step.
------------------------------------

If inside a step we want to send a metric, for example the execution time of a function or process. The step class has an attribute to access the metric produced `self.metrics_sender`.

For example we will generate a random value as a metric, and we will use the `send_metrics` method from the metrics producer class to write it into the logfile.

.. code-block:: python

  from apf.core.step import GenericStep
  import random


  class ExampleTest(GenericStep):

    def execute(self,message):
        random_value = random.random()
        self.metrics_sender.send_metrics({"random_value": random_value})

This will add a line for each execution

.. code-block::

  random_value: 0.5048194222693092
  timestamp_received: 2020-10-14 17:37:16.961150+00:00, timestamp_sent: 2020-10-14 17:37:16.961178+00:00, execution_time: 2.8e-05, candid: 1221211305915015000, ra: 145.9124706, dec: -7.018032400000001, date_mjd: 05/06/2020, 05:04:17, n_messages: 1, source: TestStep

We can send multiple metrics passing more key, value pairs to the dictionary.
