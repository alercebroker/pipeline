Profile a Step
===========================

A relevant process when developing a new step is
to check the performance and status of the step
on runtime.

Here is a quick tutorial to profile a step to check
the individual performance.

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
