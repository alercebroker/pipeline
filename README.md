[![Build Status](https://travis-ci.org/alercebroker/db-plugins.svg?branch=master)](https://travis-ci.org/alercebroker/db-plugins) [![codecov](https://codecov.io/gh/alercebroker/db-plugins/branch/master/graph/badge.svg)](https://codecov.io/gh/alercebroker/db-plugins)
# Database Plugins for ALeRCE documentation

*db_plugins* is an ORM style library created to interact with different databases. The main feature of these plugins is to provide an interface for database querying, reducing the amount of code and helping to decouple components.

# Installing *db_plugins*

*db_plugins* installation can be done with *pip*. You can clone the repository and then

```
pip install .
```

or you can install it directly from github

```
pip install git+https://github.com/alercebroker/db-plugins.git
```

# Documentation

# Database

APF Database module is a tool that will help you connect steps to a database. It works by using models that provide main operations in an abstract environment, so that you don’t have to code specific database queries and instead use python classes to interact with the database.

## 1. Plugins

APF integrates with databases through plugins. Each plugin is supposed to provide functionality for a specific database engine.

The design concept is that there are generic models that contain attributes and some methods but these are implemented in each plugin individually. For example, a PostgreSQL database will have underlying differences with a non relational database on how queries are made or on how objects are inserted, but on the higher level layer we use models with same functions so that it behaves in the same way no matter what engine is being used. This also provides the option to change database engines without having to change the step structure too much.

## 2. Database Initialization

Database plugins will read the configuration you define in `settings.py` inside STEP_CONFIG, add DB_CONFIG.

After defining DB_CONFIG you can now initiaize your database. To do so, run the `initdb` command as follows

```
apf initdb
apf make-migrations
apf migrate
```

## 3. Migrations

When changes to models are made you would want to update the database without creating it all again, or maybe you want to undo some changes and return to a previous state.

The solution is to create migrations. Migrations keep track of your database changes and let you detect differences between your database and APF models and update the database accordingly.

Migrations will be created by running `apf make_migrations`. This command will read your database credentials from DB_CONFIG inside `settings.py`.

Then, to update your database to latest changes execute `apf migrate`.

### 3.1 What migrations can and can’t detect

Migrations will detect:


* Table additions, removals.


* Column additions, removals.


* Change of nullable status on columns.


* Basic changes in indexes

Migrations can’t detect:


* Changes of table name. These will come out as an add/drop of two different tables, and should be hand-edited into a name change instead.


* Changes of column name. Like table name changes, these are detected as a column add/drop pair, which is not at all the same as a name change


# Database plugins

## 1. Generic classes


### class db_plugins.db.generic.AbstractClass()
Abstract Class model


* **Attributes**

    **name**

        the class name

    **acronym**

        the short class name


### Methods

| `get_classifications`(self)

 | Gets all classifications with the class

 |
| `get_taxonomies`(self)

      | Gets the taxonomies that the class instance belongs to

 |
<!-- !! processed by numpydoc !! -->

#### get_classifications(self)
Gets all classifications with the class

<!-- !! processed by numpydoc !! -->

#### get_taxonomies(self)
Gets the taxonomies that the class instance belongs to

<!-- !! processed by numpydoc !! -->

### class db_plugins.db.generic.AbstractTaxonomy()
Abstract Taxonomy model


* **Attributes**

    **name**

        the taxonomy name


### Methods

| `get_classes`(self)

         | Gets the classes that a taxonomy uses

                  |
| `get_classifiers`(self)

     | Gets the classifiers using the taxonomy

                |
<!-- !! processed by numpydoc !! -->

#### get_classes(self)
Gets the classes that a taxonomy uses

<!-- !! processed by numpydoc !! -->

#### get_classifiers(self)
Gets the classifiers using the taxonomy

<!-- !! processed by numpydoc !! -->

### class db_plugins.db.generic.AbstractClassifier()
Abstract Classifier model


* **Attributes**

    **name**

        name of the classifier


### Methods

| `get_classifications`(self)

 | Gets classifications done by the classifier

            |
| `get_taxonomy`(self)

        | Gets the taxonomy the classifier is using

              |
<!-- !! processed by numpydoc !! -->

#### get_classifications(self)
Gets classifications done by the classifier

<!-- !! processed by numpydoc !! -->

#### get_taxonomy(self)
Gets the taxonomy the classifier is using

<!-- !! processed by numpydoc !! -->

### class db_plugins.db.generic.AbstractAstroObject()
Abstract Object model


* **Attributes**

    **oid: str**

        object identifier

    **nobs: str**

        number of observations

    **lastmjd: float**

        date (in mjd) of the last observation

    **firstmjd: float**

        date (in mjd) of the first observation

    **meanra: float**

        mean right ascension coordinates

    **meandec: float**

        mean declination coordinates

    **sigmara: float**

        error for right ascension coordinates

    **sigmadec: float**

        error for declination coordinates

    **deltajd: float**

        difference between last and first mjd


### Methods

| `get_classifications`(self)

 | Gets all classifications on the object

                 |
| `get_detections`(self)

      | Gets all detections of the object

                      |
| `get_features`(self)

        | Gets features associated with the object

               |
| `get_lightcurve`(self)

      | Gets the lightcurve of the object

                      |
| `get_magnitude_statistics`(self)

 | Gets magnitude statistics for the object

               |
| `get_non_detections`(self)

       | Gets all non_detections of the object

                  |
| `get_xmatches`(self)

             | Gets crossmatch information for the object

             |
<!-- !! processed by numpydoc !! -->

#### get_classifications(self)
Gets all classifications on the object

<!-- !! processed by numpydoc !! -->

#### get_detections(self)
Gets all detections of the object

<!-- !! processed by numpydoc !! -->

#### get_features(self)
Gets features associated with the object

<!-- !! processed by numpydoc !! -->

#### get_lightcurve(self)
Gets the lightcurve of the object

<!-- !! processed by numpydoc !! -->

#### get_magnitude_statistics(self)
Gets magnitude statistics for the object

<!-- !! processed by numpydoc !! -->

#### get_non_detections(self)
Gets all non_detections of the object

<!-- !! processed by numpydoc !! -->

#### get_xmatches(self)
Gets crossmatch information for the object

<!-- !! processed by numpydoc !! -->

### class db_plugins.db.generic.AbstractClassification()
Abstract Classification model


* **Attributes**

    **class_name**

        name of the class

    **probability**

        probability of the classification

    **probabilities**

        probabilities for each class


### Methods

| `get_class`(self)

                | Gets the class of the classification

                   |
| `get_classifier`(self)

           | Gets the classifier used

                               |
| `get_object`(self)

               | Gets the object classifified

                           |
<!-- !! processed by numpydoc !! -->

#### get_class(self)
Gets the class of the classification

<!-- !! processed by numpydoc !! -->

#### get_classifier(self)
Gets the classifier used

<!-- !! processed by numpydoc !! -->

#### get_object(self)
Gets the object classifified

<!-- !! processed by numpydoc !! -->

### class db_plugins.db.generic.AbstractFeatures()
Abstract Features model

version

    name of the version used for features


* **Attributes**

    **version**


<!-- !! processed by numpydoc !! -->

### class db_plugins.db.generic.AbstractMagnitudeStatistics()
Abstract Magnitude Statistics model


* **Attributes**

    **magnitude_type**

        the type of the magnitude, could be psf or ap

    **fid**

        magnitude band identifier, 1 for red, 2 for green

    **mean**

        mean magnitude meassured

    **median**

        median of the magnitude

    **max_mag**

        maximum value of magnitude meassurements

    **min_mag**

        minimum value of magnitude meassurements

    **sigma**

        error of magnitude meassurements

    **last**

        value for the last magnitude meassured

    **first**

        value for the first magnitude meassured


### Methods

| `get_object`(self)

               | Gets the object associated with the stats

              |
<!-- !! processed by numpydoc !! -->

#### get_object(self)
Gets the object associated with the stats

<!-- !! processed by numpydoc !! -->

### class db_plugins.db.generic.AbstractNonDetection()
Abstract model for non detections


* **Attributes**

    **mjd**

        date of the non detection in mjd

    **diffmaglim: float**

        magnitude of the non detection

    **fid**

        band identifier 1 for red, 2 for green


### Methods

| `get_object`(self)

               | Gets the object related

                                |
<!-- !! processed by numpydoc !! -->

#### get_object(self)
Gets the object related

<!-- !! processed by numpydoc !! -->

### class db_plugins.db.generic.AbstractDetection()
Abstract model for detections


* **Attributes**

    **candid**

        candidate identifier

    **mjd**

        date of the detection in mjd

    **fid**

        band identifier, 1 for red, 2 for green

    **ra**

        right ascension coordinates

    **dec**

        declination coordinates

    **rb**

        real bogus

    **magap**

        ap magnitude

    **magpsf**

        psf magnitude

    **sigmapsf**

        error for psf magnitude

    **sigmagap**

        error for ap magnitude

    **magpsf_corr**

        magnitude correction for magpsf

    **magap_corr**

        magnitude correction for magap

    **sigmapsf_corr**

        correction for sigmapsf

    **sigmagap_corr**

        correction for sigmagap

    **avro**

        url for avro file in s3


### Methods

| `get_object`(self)

               | Gets the object related

                                |
<!-- !! processed by numpydoc !! -->

#### get_object(self)
Gets the object related

<!-- !! processed by numpydoc !! -->
## 2. SQL

### Initialize database

Before you connect to your database, make sure you initialize it first.
To do that execute the following command from your step root folder

`apf initdb`

When you run this command with an empty database it will create the
following schema:



![Diagram](docs/source/_static/images/diagram.png)

### Migrations

Migrations keep track of database changes. To fully init the database with your
step configuration run

```
apf make_migrations
apf migrate
```

This will set the head state for tracking changes on the database and also execute any migrations that might be present.

The first command `apf make_migrations` will create migration files according to differences from apf models and your database.

The seccond command `apf migrate` will execute the migrations and update your database.

### Set database Connection

```
from db_plugins.db.sql import DatabaseConnection
from db_plugins.db.sql.models import *
```

```
db_config = {
    "SQL": "sqlite:///:memory:"
}
```

The URL used here follows this format: dialect[+driver]://user:password@host/dbname[?key=value..]

```
db = DatabaseConnection()
db.init_app(db_config["SQL"], Base)
db.create_session()
```

The above code will create a connection to the database wich
we will later use to store objects.

### Create model instances

Use get_or_create function to get an instance of a model. The instance
will be an object from the database if it already exists or it will
create a new instance. **This object is not yet added to the database**

```
instance, created = db.get_or_create(Model,args)
```

```
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
```

```
obj, created = db.get_or_create(AstroObject, **model_args)
print(obj, "created: " + str(created))
```

`<AstroObject(oid='ZTFid')> created: False`

### Add related models

Lets say for example that we want to create a class that belongs to a
taxonomy.

```
class_, created = db.get_or_create(Class, name="Super Nova", acronym="SN")
class_
```

`<Class(name='Super Nova', acronym='SN')>`

```
taxonomy, created = db.get_or_create(Taxonomy, name="Example")
print(taxonomy, "created: " + str(created))
class_.taxonomies.append(taxonomy)
```

`<Taxonomy(name='Example')> created: False`

```
class_.taxonomies
```

`[<Taxonomy(name='Example')>, <Taxonomy(name='Example')>]`

```
taxonomy.classes
```

`[<Class(name='Super Nova', acronym='SN')>]`

As you can see, adding a model works both sides.

When we add a taxonomy to a class it also means that a class is added to
the taxonomy.

### Add objects to the database

All our instanced objects are not yet added to the database. To do that
we use `session.add` or `session.add_all` methods

```
db.session.add(class_)
db.session.commit()
```

### DatabaseConnection documentation


### class db_plugins.db.sql.DatabaseConnection(db_credentials=None, base=None)
### Methods

| `bulk_insert`(self, objects, model)

 | Inserts multiple objects to the database improving performance

 |
| `check_exists`(self, model, filter_by)

 | Check if record exists in database.

                            |
| `get_or_create`(self, model[, filter_by])

 | Initializes a model by creating it or getting it from the database if it exists

 |
| `update`(self, instance, args)

            | Updates an object

                                                               |
| **cleanup**

                                 |                                                                                 |
| **create_db**

                               |                                                                                 |
| **create_scoped_session**

                   |                                                                                 |
| **create_session**

                          |                                                                                 |
| **drop_db**

                                 |                                                                                 |
| **init_app**

                                |                                                                                 |
| **query**

                                   |                                                                                 |
<!-- !! processed by numpydoc !! -->

