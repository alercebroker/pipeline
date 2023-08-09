.. ALeRCE Database Plugins documentation master file, created by
   sphinx-quickstart on Tue Dec  3 11:25:36 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Database Models for ALeRCE
==========================

This lib offers database initialization procedures for MongoDB and PostgreSQL, using the ALeRCE core models.

Installing *db_plugins*
=======================

For local development:

*db_plugins* installation is recommended with *poetry*. You can clone the repository and then

.. code-block:: console

   cd libs/db-plugins
   poetry install

Alternatively, you can use pip

.. code-block:: console

   cd libs/db-plugins
   pip install -e .

As a local dependency:

.. code-block:: console

   cd some_package
   poetry add ../libs/db-plugins

As a remote dependency
.. code-block:: console

   poetry add "https://github.com/alercebroker/db-plugins.git#subdirectory=libs/db-plugins"

Documentation
=============

.. toctree::
  database
  db_plugins
  :maxdepth: 3
