Database Connection
=============

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


2. Migrations
-------------
When changes to models are made you would want to update the database without creating it all again, or maybe you want to undo some changes and return to a previous state.

The solution is to create migrations. Migrations keep track of your database changes and let you detect differences between your database and models and update the database accordingly.

Migrations will be created by running ``dbp make_migrations``. This command will read your database credentials from `DB_CONFIG` inside ``settings.py``.

Then, to update your database to latest changes execute ``dbp migrate``.


