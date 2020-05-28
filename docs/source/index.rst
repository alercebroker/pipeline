.. ALeRCE Database Plugins documentation master file, created by
   sphinx-quickstart on Tue Dec  3 11:25:36 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Database Plugins for ALeRCE documentation
=============================================================

*db_plugins* is an ORM style library created to interact with different databases. The main feature of these plugins is to provide an interface for database querying, reducing the amount of code and helping to decouple components.

Installing *db_plugins*
====================

*db_plugins* installation can be done with *pip*. You can clone the repository and then

.. code-block:: bash

      pip install .

or you can install it directly from github

.. code-block:: bash

    pip install git+https://github.com/alercebroker/db-plugins.git

Documentation
=============

.. toctree::
  new_step
  database
  profiling
  core
  consumers
  db_plugins
  producers
  db
  :maxdepth: 2
