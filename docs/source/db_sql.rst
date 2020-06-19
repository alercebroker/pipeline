Initialize database
++++++++++++++++++++
Before you connect to your database, make sure you initialize it first.
To do that execute the following command from your step root folder

``apf initdb``

When you run this command with an empty database it will create the
following schema:

.. .. image:: _static/images/diagram.png
..     :align: center

Migrations
+++++++++++++++
Migrations keep track of database changes. To fully init the database with your
step configuration run

.. code:: python

    apf make_migrations
    apf migrate


This will set the head state for tracking changes on the database and also execute any migrations that might be present.

The first command ``apf make_migrations`` will create migration files according to differences from apf models and your database.

The seccond command ``apf migrate`` will execute the migrations and update your database.

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

    from db_plugins.db import SQLDatabase
    from db_plugins.db.sql.models import *

.. code:: ipython3

    db_config = {
        "SQL": "sqlite:///:memory:"
    }

The URL used here follows this format: `dialect[+driver]://user:password@host/dbname[?key=value..]`

.. code:: ipython3

    db = SQLDatabase()
    db.connect(config=db_config)

The above code will create a connection to the database wich
we will later use to store objects.

Create model instances
+++++++++++++++++++++++

Use get_or_create function to get an instance of a model. The instance
will be an object from the database if it already exists or it will
create a new instance. **This object is not yet added to the database**

.. code:: python

   instance, created = db.session.query().get_or_create(Model,args)

.. code:: ipython3

    model_args = {
        "oid":"ZTFid",
        "nobs":1,
        "lastmjd":1,
        "meanra":1,
        "meandec":1,
        "sigmara":1,
        "sigmadec":1,
        "deltajd":1,
        "firstmjd":1
    }

.. code:: ipython3

    obj, created = db.session.query().get_or_create(AstroObject, **model_args)
    print(obj, "created: " + str(created))

``<AstroObject(oid='ZTFid')> created: False``


Add related models
++++++++++++++++++

Lets say for example that we want to create a class that belongs to a
taxonomy.

.. code:: ipython3

    class_, created = db.session.query().get_or_create(Class, name="Super Nova", acronym="SN")
    class_

``<Class(name='Super Nova', acronym='SN')>``



.. code:: ipython3

    taxonomy, created = db.session.query().get_or_create(Taxonomy, name="Example")
    print(taxonomy, "created: " + str(created))
    class_.taxonomies.append(taxonomy)

``<Taxonomy(name='Example')> created: False``

.. code:: ipython3

    class_.taxonomies

``[<Taxonomy(name='Example')>, <Taxonomy(name='Example')>]``


.. code:: ipython3

    taxonomy.classes

``[<Class(name='Super Nova', acronym='SN')>]``



As you can see, adding a model works both sides.

When we add a taxonomy to a class it also means that a class is added to
the taxonomy.

Add objects to the database
++++++++++++++++++++++++++++

All our instanced objects are not yet added to the database. To do that
we use ``session.add`` or ``session.add_all`` methods

.. code:: ipython3

    db.session.add(class_)
    db.session.commit()


DatabaseConnection documentation
+++++++++++++++++

.. autoclass:: db_plugins.db SQLDatabase

