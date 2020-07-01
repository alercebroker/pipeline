Database
=============
ALeRCE Database module is a tool that will help you connect services like APIs and pipeline steps to a database. It works by using models that provide main operations in an abstract environment, so that you don't have to code specific database queries and instead use python classes to interact with the database.


1. Plugins
------------
ALeRCE integrates with databases through plugins. Each plugin is supposed to provide functionality for a specific database engine.

The design concept is that there are generic models that contain attributes and some methods but these are implemented in each plugin individually. For example, a PostgreSQL database will have underlying differences with a non relational database on how queries are made or on how objects are inserted, but on the higher level layer we use models with same functions so that it behaves in the same way no matter what engine is being used. This also provides the option to change database engines without having to change the application structure too much.

2. Database Initialization
--------------------------
Database plugins will read the configuration you define in a ``settings.py`` file. This file should have a `DB_CONFIG` dictionary with the database connection parameters.

Here is and example on the params used with the SQL plugin:

.. code-block:: python

    DB_CONFIG: {
        "SQL": {
            "SQLALCHEMY_DATABASE_URL": "sqlite:///:memory:"
        }
    }

After defining `DB_CONFIG` you can now initiaize your database. To do so, run the ``initdb`` command as follows


.. code-block:: bash

    dbp initdb


3. Migrations
-------------
When changes to models are made you would want to update the database without creating it all again, or maybe you want to undo some changes and return to a previous state.

The solution is to create migrations. Migrations keep track of your database changes and let you detect differences between your database and models and update the database accordingly.

Migrations will be created by running ``dbp make_migrations``. This command will read your database credentials from `DB_CONFIG` inside ``settings.py``.

Then, to update your database to latest changes execute ``dbp migrate``.


