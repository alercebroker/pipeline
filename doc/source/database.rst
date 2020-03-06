Database
=============
APF Database module is a tool that will help you connect steps to a database. It works by using models that provide main operations in an abstract environment, so that you don't have to code specific database queries and instead use python classes to interact with the database.


1. Plugins
------------
APF integrates with databases through plugins. Each plugin is supposed to provide functionality for a specific database engine.

The design concept is that there are generic models that contain attributes and some methods but these are implemented in each plugin individually. For example, a PostgreSQL database will have underlying differences with a non relational database on how queries are made or on how objects are inserted, but on the higher level layer we use models with same functions so that it behaves in the same way no matter what engine is being used. This also provides the option to change database engines without having to change the step structure too much.

2. Database Initialization
--------------------------
Database plugins will read the configuration you define in ``settings.py`` inside `STEP_CONFIG`, add `DB_CONFIG`.

After defining `DB_CONFIG` you can now initiaize your database. To do so, run the ``initdb`` command as follows


.. code-block:: bash

    apf initdb
    apf make_migrations
    apf migrate


3. Migrations
-------------
When changes to models are made you would want to update the database without creating it all again, or maybe you want to undo some changes and return to a previous state.

The solution is to create migrations. Migrations keep track of your database changes and let you detect differences between your database and APF models and update the database accordingly.

Migrations will be created by running ``apf make_migrations``. This command will read your database credentials from `DB_CONFIG` inside ``settings.py``.

Then, to update your database to latest changes execute ``apf migrate``.

3.1 What migrations can and can't detect
+++++++++++++++++++++++++++++++++++++++++
Migrations will detect:

- Table additions, removals.

- Column additions, removals.

- Change of nullable status on columns.

- Basic changes in indexes

Migrations can't detect:

- Changes of table name. These will come out as an add/drop of two different tables, and should be hand-edited into a name change instead.

- Changes of column name. Like table name changes, these are detected as a column add/drop pair, which is not at all the same as a name change.
