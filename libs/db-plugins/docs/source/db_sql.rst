Initialize database
++++++++++++++++++++
Before you connect to your database, make sure you initialize it first.
To do that execute the following command from a directory containing a ``settings.py`` file.

You can also privide the settings directory with ``--settings_path``.

.. code:: bash

   dbp initdb

When you run this command with an empty database it will create the
following schema:

.. image:: _static/images/diagram.png
    :align: center

Migrations
+++++++++++++++
Migrations keep track of database changes. To fully init the database with your
step configuration run

.. code:: python

    dbp make_migrations
    dbp migrate


This will set the head state for tracking changes on the database and also execute any migrations that might be present.

The first command ``dbp make_migrations`` will create migration files according to differences from dbp models and your database.

The seccond command ``dbp migrate`` will execute the migrations and update your database.

What migrations can and can't detect
+++++++++++++++++++++++++++++++++++++++++
Migrations will detect:

- Table additions, removals.

- Column additions, removals.

- Change of nullable status on columns.

- Basic changes in indexes

Migrations can't detect:

- Changes of table name. These will come out as an add/drop of two different tables, and should be hand-edited into a name change instead.

- Changes of column name. Like table name changes, these are detected as a column add/drop pair, which is not at all the same as a name change.

Set database Connection
++++++++++++++++++++++++

.. code:: ipython3

   from db_plugins.db.sql import SQLConnection
   from db_plugins.db.sql.models import *

.. code:: ipython3

    db_config = {
        "SQL": {
          "ENGINE": "postgresql",
          "HOST": "host",
          "USER": "username",
          "PASSWORD": "pwd",
          "PORT": 5432, # postgresql tipically runs on port 5432. Notice that we use an int here.
          "DB_NAME": "database",
        }
    }

.. code:: ipython3

    db = SQLConnection()
    db.connect(config=db_config["SQL"])

The above code will create a connection to the database wich
we will later use to store objects.

Create model instances
+++++++++++++++++++++++

Use get_or_create function to get an instance of a model. The instance
will be an object from the database if it already exists or it will
create a new instance.

.. code:: ipython3

    oid = "ZTF_OID"
    model_args = {}
    model_args["ndethist"] = 0
    model_args["ncovhist"] = 0
    model_args["mjdstarthist"] = 0.0
    model_args["mjdendhist"] = 0.0
    model_args["firstmjd"] = 0.0
    model_args["lastmjd"] = 0.0
    model_args["ndet"] = 0
    model_args["deltajd"] = 0.0
    model_args["meanra"] = 0.0
    model_args["meandec"] = 0.0
    model_args["step_id_corr"] = "v1.0.0"
    model_args["corrected"] = False
    model_args["stellar"] = False

.. code:: ipython3

    obj, created = db.query(Object).get_or_create(filter_by={"oid": oid}, **model_args)
    print(obj, "created: " + str(created))

``<AstroObject(oid='ZTFid')> created: False``

In the above example we use the object id as a filter since it is the primary key of the Object model. Notice that ``get_or_create`` can receive the model as a parameter or it can inherit it from the ``query`` parameter.

The **important** part is that ``model_args`` should contain all attributes of the table.

Add multiple objects to the database
++++++++++++++++++++++++++++

If you need to insert multiple objects at once, there is a faster way than using ``get_or_create`` multiple times. You can use the ``bulk_insert`` method.

It will take a list of dictionaries where each dictionary has all the attributes for a table as keys.

.. code:: ipython3

   db.query(Detection).bulk_insert(prv_detections)


Where prv_detections is a list of dict where each dict contains information to populate Detection table.

Update instances
++++++++++++++++

There is a particularity when you make updates to instances. Let's say that we have an object instance and we want to change its lastmjd.

.. code:: ipython3

   obj = db.query(Object).get_or_create(filter_by={"oid": "ZTF123"})

   obj = db.query().update(obj, {"lastmjd": 12345})

After updating the instance you have to commit the changes. This is done in the following way:

.. code:: python

   db.session.commit()

DatabaseConnection documentation
++++++++++++++++++++++++++++++++

.. autoclass:: db_plugins.db.sql.SQLConnection
    :members:

