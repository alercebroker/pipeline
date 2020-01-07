Profile and monitor a Step
===========================

A relevant process when developing a new step is
to check the performance and status of the step
on runtime.

Here is a quick tutorial to profile a step to check
the individual performance and monitor the step running
on a more scalable environment.

Step Profiling
--------------

The easiest way to profile a step is using *cProfile*, for this
we just have to run the step with the following command:

.. code-block:: bash

  python -m cProfile -o <outputfile> scripts/run_step.py


This command has to be run with the `run_step.py` script, using multiprocessing
is not easy to profile.

Depending on the consumer the step must be stopped manually with keyboard interrupt
(`ctrl + c`) and a file will be generated.

There are several tools to visually inspect the output file. The recommended one
is `snakeviz <https://jiffyclub.github.io/snakeviz/>`_. Using `snakeviz` is fairly simple
just run

.. code-block:: bash

  snakeviz <outputfile>

This will prompt a web browser with the profiling, it is recommended to lower the depth of
the plot to 3 or 5 for a faster load.

Step Monitoring
---------------

Each step has by default a method to send metrics into an ElasticSearch (ES) Cluster.

The :meth:`~apf.core.GenericStep.send_metrics` method is called after
each message is processed and sends the execution time to the ES cluster. This allows
the developer and other users to visualize the performance of each step.

To start sending metrics and visualizing them on Kibana and Grafana follow this steps:

1.- **(For develop only)** Create a ES Cluster and connect to Kibana and Grafana:
    We use a `docker-compose.yml` file to create a develop environment, an example is the following:

    .. code-block:: yaml

      version: '3'
        services:
        elasticsearch:
        image: "elasticsearch:7.5.0"
        ports:
          - "9200:9200"
          - "9300:9300"
        environment:
          - "discovery.type=single-node"
        volumes:
          - "./elasticsearch.yml:/usr/share/elasticsearch/config/elasticsearch.yml"
        grafana:
        image: "grafana/grafana"
        ports:
          - "3000:3000"
        links:
          - elasticsearch
        kibana:
        image: "kibana:7.5.0"
        ports:
          - "5601:5601"
        links:
          - elasticsearch

    We also use a configuration file for ElasticSearch:

    .. code-block:: yaml

      cluster.name: "docker-cluster"
      network.host: 0.0.0.0
      http.cors.enabled: true
      http.cors.allow-origin: "*"

    This allows Grafana to access the ElasticSearch Data.

    Running `docker-compose up -d` will create the environment with Grafana running on **localhost:3000**, Kibana on **localhost:5601** and ElasticSearch on **localhost:9200**.

    Now we just have to configure Grafana and Kibana using `http://elasticsearch:9200` as the elasticsearch HTTP API.

2.- After creating an environment to monitor the step we need to configure it to send metrics. So we need to add the **ES_CONFIG** variable to the `settings.py` file.

     .. code-block:: python

       #settings.py
       ES_CONFIG = {'INDEX_PREFIX':'test_step'}

       STEP_CONFIG = {
            ...
            "ES_CONFIG":ES_CONFIG
            }

     The **INDEX_PREFIX** variable is used to create the index for ES. The index used to store metrics is **INDEX_PREFIX-ClassName-Date** and each document has two default variables:

      a. **source**: ClassName of the step sending the metric.
      b. **@timestamp**: DateTime in UTC when the metric is created.

3.- Create a new visualization on Grafana. For this we first gonna configure a new data source.


    .. image:: _static/images/datasource.png
        :align: center

    In here we want to add an ElasticSearch datasource with the following

    .. image:: _static/images/es.png
        :align: center

    Using the `docker-compose` file we have the following parameters:

      a. **URL**: `http://elasticsearch:9200`
      b. **index_name**: `test_step*` or the **INDEX_PREFIX** used.
      c. **Time field name**: `@timestamp` using the default field from :meth:`~apf.core.GenericStep.send_metrics`.

    Then we can create a new Dashboard. An example of a panel configuration is the following:

    For the Query we want all documents that have `execution_time` a default metric sended automatically by each step.

    .. image:: _static/images/query.png
        :align: center

    we are using `execution_time: *` as the query and adding `Group by: Terms: source.keyword` and `Then by: Date Histogram: @timestamp`.

    This will generate the following plot when we run a single or multiple process step.


    .. image:: _static/images/graph.png
        :align: center
