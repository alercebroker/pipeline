Initialize database
+++++++++++++++++++

Before you connect to your database, make sure you initialize it first.
To do that execute the following command from your step root folder

.. code-block:: console

   dbp initdb

When you run this command with an empty database it will create the
schema defined by SQLAlchemy models:

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
