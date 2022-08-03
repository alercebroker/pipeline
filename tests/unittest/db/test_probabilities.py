from db_plugins.db.mongo.models import Object
import unittest
import mongomock

"""
This test the object creation (not the main objetive)
and explain how to query and update the objects with 
probabilities (main focus of the tests)
"""

class MongoProbabilitiesTest(unittest.TestCase):
    def setUp(self):
        client = mongomock.MongoClient()
        self.database = client["database"]
        self.obj_collection = self.database["object"]
        #self.obj_collection.insert_one({"test": "test"})
        """
        self.mongo_query_class = mongo_query_creator(
            mongomock.collection.Collection
        )
        self.query = self.mongo_query_class(
            model=Object,
            database=self.database,
            _db_store=self.database._store,
        )
        """
    

    def create_2_objects(self):
        model_1 = Object(
            aid="aid1",
            oid="oid1",
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra=100.0,
            meandec=50.0,
            ndet=2,
            probabilities=[
                {
                    "classifier_name": "stamp_classifier",
                    "classifier_version": "stamp_classifier_1.0.0",
                    "class_name": "CLASS1",
                    "probability": 0.6,
                    "ranking": 1
                },
                {
                    "classifier_name": "stamp_classifier",
                    "classifier_version": "stamp_classifier_1.0.0",
                    "class_name": "CLASS2",
                    "probability": 0.4,
                    "ranking": 1
                }
            ]
        )
        model_2 = Object(
            aid="aid2",
            oid="oid2",
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra=100.0,
            meandec=50.0,
            ndet=5,
            probabilities=[
                {
                    "classifier_name": "stamp_classifier",
                    "classifier_version": "stamp_classifier_1.0.0",
                    "class_name": "CLASS1",
                    "probability": 0.4,
                    "ranking": 1
                },
                {
                    "classifier_name": "stamp_classifier",
                    "classifier_version": "stamp_classifier_1.0.0",
                    "class_name": "CLASS2",
                    "probability": 0.6,
                    "ranking": 1
                }
            ]
        )        
        self.obj_collection.insert_many([model_1, model_2])

    def test_query_with_probabilities_find_one(self):
        self.create_2_objects()
        #find object 1
        f = self.obj_collection.find(
            {
                "probabilities": {
                    "$elemMatch": {
                        "classifier_name": "stamp_classifier",
                        "classifier_version": "stamp_classifier_1.0.0",
                        "class_name": "CLASS1",
                        "probability": {"$gte": 0.5},
                        "ranking": 1
                    }
                }
            }
        )
        fl = list(f)
        self.assertIsNotNone(f)
        self.assertEqual(len(fl), 1)
        self.assertEqual(fl[0]["aid"], "aid1")
        
        # find object 2
        f = self.obj_collection.find(
            {
                "probabilities": {
                    "$elemMatch": {
                        "classifier_name": "stamp_classifier",
                        "classifier_version": "stamp_classifier_1.0.0",
                        "class_name": "CLASS2",
                        "probability": {"$gte": 0.5},
                        "ranking": 1
                    }
                }
            }
        )
        fl = list(f)
        self.assertIsNotNone(f)
        self.assertEqual(len(fl), 1)
        self.assertEqual(fl[0]["aid"], "aid2")

    def test_query_with_probabilities_find_none(self):
        self.create_2_objects()

        f = self.obj_collection.find(
            {
                "probabilities": {
                    "$elemMatch": {
                        "classifier_name": "stamp_classifier",
                        "classifier_version": "stamp_classifier_1.0.0",
                        "class_name": "CLASS1",
                        "probability": {"$gte": 1},
                        "ranking": 1
                    }
                }
            }
        )
        fl = list(f)
        self.assertIsNotNone(f)
        self.assertEqual(len(fl), 0)

    def test_query_with_probabilities_find_two(self):
        self.create_2_objects()

        f = self.obj_collection.find(
            {
                "probabilities": {
                    "$elemMatch": {
                        "classifier_name": "stamp_classifier",
                        "classifier_version": "stamp_classifier_1.0.0",
                        "class_name": "CLASS2",
                        "probability": {"$gte": 0.1},
                        "ranking": 1
                    }
                }
            }
        )
        fl = list(f)
        self.assertIsNotNone(f)
        self.assertEqual(len(fl), 2)


    def test_update_probability(self):
        self.create_2_objects()

        self.obj_collection.update_one(
            {
                "aid": "aid1",
                "probabilities":{
                    "$elemMatch" : {
                        "classifier_name": "stamp_classifier",
                        "classifier_version": "stamp_classifier_1.0.0",
                        "class_name": "CLASS1"
                    }
                }
            },
            {
                "$set": {"probabilities.$.probability": 1.0}
                
            }
        )

        f1 = self.obj_collection.find_one({"aid": "aid1"})
        print(f"objeto 1 {f1}")
        expected_object_1_probabilities = [
            {
                "classifier_name": "stamp_classifier",
                "classifier_version": "stamp_classifier_1.0.0",
                "class_name": "CLASS1",
                "probability": 1,
                "ranking": 1
            },
            {
                "classifier_name": "stamp_classifier",
                "classifier_version": "stamp_classifier_1.0.0",
                "class_name": "CLASS2",
                "probability": 0.4,
                "ranking": 1
            }
        ]
        self.assertEqual(f1["probabilities"], expected_object_1_probabilities)

        f2 = self.obj_collection.find_one({"aid": "aid2"})
        print(f"objeto 2 {f2}")
        expected_object_2_probabilities = [
            {
                "classifier_name": "stamp_classifier",
                "classifier_version": "stamp_classifier_1.0.0",
                "class_name": "CLASS1",
                "probability": 0.4,
                "ranking": 1
            },
            {
                "classifier_name": "stamp_classifier",
                "classifier_version": "stamp_classifier_1.0.0",
                "class_name": "CLASS2",
                "probability": 0.6,
                "ranking": 1
            }
        ]
        self.assertEqual(f2["probabilities"], expected_object_2_probabilities)
