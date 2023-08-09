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


Database
========
1. Database Initialization
--------------------------
Database plugins will read the configuration you define in a ``settings.py`` file. This file should have a `DB_CONFIG` dictionary with the database connection parameters.

Here is and example on the params used with the SQL plugin:

.. code-block:: python

   DB_CONFIG: {
       "SQL": {
           "SQLALCHEMY_DATABASE_URL": "sqlite:///:memory:"
       }
   }

After defining `DB_CONFIG` you can now initialize your database. To do so, run the ``initdb`` command as follows


.. code-block:: console

   dbp initdb


2. Migrations (SQL only)
------------------------
When changes to models are made you would want to update the database without creating it all again, or maybe you want to undo some changes and return to a previous state.

The solution is to create migrations. Migrations keep track of your database changes and let you detect differences between your database and models and update the database accordingly.

Migrations will be created by running ``dbp make_migrations``. This command will read your database credentials from `DB_CONFIG` inside ``settings.py``.

Then, to update your database to latest changes execute ``dbp migrate``.

Database plugins
================

1. SQL
------

Initialize database
+++++++++++++++++++

Before you connect to your database, make sure you initialize it first.
To do that execute the following command from your step root folder

.. code-block:: console

   dbp initdb

When you run this command with an empty database it will create the
following schema:

.. image:: docs/source/_static/images/diagram.png
   :align: center

Migrations
++++++++++

Migrations keep track of database changes. To fully initialize the database with your
step configuration run

.. code-block:: python

   dbp make_migrations
   dbp migrate


This will set the head state for tracking changes on the database and also execute any migrations that might be present.

The first command ``dbp make_migrations`` will create migration files according to differences from dbp models and your database.

The seccond command ``dbp migrate`` will execute the migrations and update your database.

What migrations can and can't detect
++++++++++++++++++++++++++++++++++++

Migrations will detect:

- Table additions, removals.

- Column additions, removals.

- Change of nullable status on columns.

- Basic changes in indexes

Migrations can't detect:

- Changes of table name. These will come out as an add/drop of two different tables, and should be hand-edited into a name change instead.

- Changes of column name. Like table name changes, these are detected as a column add/drop pair, which is not at all the same as a name change.

