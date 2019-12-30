Database
=============
APF Database module is a tool that will help you connect steps to a database. It works by using models that provide main operations in an abstract environment, so that you don't have to code specific database queries and instead use python classes to interact with the database.


1. Plugins
------------
APF integrates with databases through plugins. Each plugin is supposed to provide functionality for a specific database engine.

The design concept is that there are generic models that contain attributes and some methods but these are implemented in each plugin individually. For example, a PostgreSQL database will have underlying differences with a non relational database on how queries are made or on how objects are inserted, but on the higher level layer we use models with same functions so that it behaves in the same way no matter what engine is being used. This also provides the option to change database engines without having to change the step structure too much. 

2. Database Initialization
--------------------------
Database plugins will read the configuration you define in the settings.py file of the step. In ``settings.py`` inside `STEP_CONFIG`, add `DB_CONFIG` wich should contain your database connection information.

After defining `DB_CONFIG` you can now initiaize your database. To do so, run the `initdb`command as follows

3. Migrations
-------------
When you start adding changes to models you would want to update the database without creating it all again, or maybe you want to undo some changes and return to a previous state.

The solution is to create migrations. Migrations keep track of your database changes and let you detect differences between your database and APF models and update the database accordingly.

When creating a step, 


