var db = connect("mongodb://root:root@localhost:27017/admin");

db = db.getSiblingDB('develop'); // we can not use "use" statement here to switch db

db.createUser(
    {
        user: "poolento",
        pwd: "poolento",
        roles: [ { role: "readWrite", db: "develop"} ],
        passwordDigestor: "server",
    }
)