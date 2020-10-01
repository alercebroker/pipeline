Database Connection
=============
*db_plugins* is an ORM style library created to interact with different databases. The main feature of these plugins is to provide an interface for database querying, reducing the amount of code and helping to decouple components.


1. Plugins
------------
ALeRCE integrates with databases through plugins. Each plugin is supposed to provide functionality for a specific database engine.

The design concept is that there are multiple database connections but all of them share the same interface so that connecting to any provided engine is done in a similar way.

This provides a way to connect and query different database engines using the same methods and classes, for example a database connection `db` has a `query` method that returns a `BaseQuery` object that has methods for inserting, updating or getting paginated results from a SQL database, but it also works the same way for a Mongodb database.

This also provides the option to change database engines without having to change the application structure too much.

2. Database Initialization
--------------------------
Database plugins will read the configuration you define in a ``settings.py`` file. This file should have a `DB_CONFIG` dictionary with the database connection parameters.

Here is and example on the params used with the SQL plugin:

.. code-block:: python

    DB_CONFIG: {
        "SQL": {
          "ENGINE": "postgresql",
          "HOST": "host",
          "USER": "username",
          "PASSWORD": "pwd",
          "PORT": 5432, # postgresql tipically runs on port 5432. Notice that we use an int here.
          "DB_NAME": "database",
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


