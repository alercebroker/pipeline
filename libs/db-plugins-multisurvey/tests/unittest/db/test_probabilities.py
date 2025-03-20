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

    def create_2_objects(self):
        model_1 = Object(
            aid="aid1",
            oid="oid1",
            tid="tid1",
            sid="sid",
            corrected=True,
            stellar=True,
            sigmara=0.1,
            sigmadec=0.1,
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            deltajd=1,
            meanra=100.0,
            meandec=50.0,
            ndet=2,
            probabilities=[
                {
                    "classifier_name": "stamp_classifier",
                    "classifier_version": "stamp_classifier_1.0.0",
                    "class_name": "CLASS1",
                    "probability": 0.6,
                    "ranking": 1,
                },
                {
                    "classifier_name": "stamp_classifier",
                    "classifier_version": "stamp_classifier_1.0.0",
                    "class_name": "CLASS2",
                    "probability": 0.4,
                    "ranking": 1,
                },
            ],
        )
        model_2 = Object(
            aid="aid2",
            oid="oid2",
            sid="sid",
            tid="tid2",
            corrected=True,
            stellar=True,
            sigmara=0.1,
            sigmadec=0.1,
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            deltajd=1,
            meanra=100.0,
            meandec=50.0,
            ndet=5,
            probabilities=[
                {
                    "classifier_name": "stamp_classifier",
                    "classifier_version": "stamp_classifier_1.0.0",
                    "class_name": "CLASS1",
                    "probability": 0.4,
                    "ranking": 1,
                },
                {
                    "classifier_name": "stamp_classifier",
                    "classifier_version": "stamp_classifier_1.0.0",
                    "class_name": "CLASS2",
                    "probability": 0.6,
                    "ranking": 1,
                },
            ],
        )
        self.obj_collection.insert_many([model_1, model_2])

    def create_simple_object(self):
        model_2 = Object(
            aid="aid3",
            oid="oid3",
            sid="sid",
            tid="tid3",
            corrected=True,
            stellar=True,
            sigmara=0.1,
            sigmadec=0.1,
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            deltajd=1,
            meanra=100.0,
            meandec=50.0,
            ndet=5,
            probabilities=[],
        )
        self.obj_collection.insert_many([model_2])

    def test_query_with_probabilities_find_one(self):
        self.create_2_objects()
        # find object 1
        f = self.obj_collection.find(
            {
                "probabilities": {
                    "$elemMatch": {
                        "classifier_name": "stamp_classifier",
                        "classifier_version": "stamp_classifier_1.0.0",
                        "class_name": "CLASS1",
                        "probability": {"$gte": 0.5},
                        "ranking": 1,
                    }
                }
            }
        )
        fl = list(f)
        self.assertIsNotNone(f)
        self.assertEqual(len(fl), 1)
        self.assertEqual(fl[0]["_id"], "oid1")

        # find object 2
        f = self.obj_collection.find(
            {
                "probabilities": {
                    "$elemMatch": {
                        "classifier_name": "stamp_classifier",
                        "classifier_version": "stamp_classifier_1.0.0",
                        "class_name": "CLASS2",
                        "probability": {"$gte": 0.5},
                        "ranking": 1,
                    }
                }
            }
        )
        fl = list(f)
        self.assertIsNotNone(f)
        self.assertEqual(len(fl), 1)
        self.assertEqual(fl[0]["_id"], "oid2")

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
                        "ranking": 1,
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
                        "ranking": 1,
                    }
                }
            }
        )
        fl = list(f)
        self.assertIsNotNone(f)
        self.assertEqual(len(fl), 2)

    def test_update_probability(self):
        """
        According to my documentation reading, to update an object we must first check if the
        probability we desire to update exist (a filter for the oid and a elemMatch for the probabilitie).
        If it does we can update it, if it doesn't we need to push it.
        """
        self.create_2_objects()

        self.obj_collection.update_one(
            {
                "_id": "oid1",
                "probabilities": {
                    "$elemMatch": {
                        "classifier_name": "stamp_classifier",
                        "classifier_version": "stamp_classifier_1.0.0",
                        "class_name": "CLASS1",
                    }
                },
            },
            {"$set": {"probabilities.$.probability": 1.0}},
        )

        f1 = self.obj_collection.find_one({"_id": "oid1"})
        expected_object_1_probabilities = [
            {
                "classifier_name": "stamp_classifier",
                "classifier_version": "stamp_classifier_1.0.0",
                "class_name": "CLASS1",
                "probability": 1,
                "ranking": 1,
            },
            {
                "classifier_name": "stamp_classifier",
                "classifier_version": "stamp_classifier_1.0.0",
                "class_name": "CLASS2",
                "probability": 0.4,
                "ranking": 1,
            },
        ]
        self.assertEqual(f1["probabilities"], expected_object_1_probabilities)

        f2 = self.obj_collection.find_one({"_id": "oid2"})
        expected_object_2_probabilities = [
            {
                "classifier_name": "stamp_classifier",
                "classifier_version": "stamp_classifier_1.0.0",
                "class_name": "CLASS1",
                "probability": 0.4,
                "ranking": 1,
            },
            {
                "classifier_name": "stamp_classifier",
                "classifier_version": "stamp_classifier_1.0.0",
                "class_name": "CLASS2",
                "probability": 0.6,
                "ranking": 1,
            },
        ]
        self.assertEqual(f2["probabilities"], expected_object_2_probabilities)

    def test_insert_probability(self):
        """
        An example of a probabilitie push
        """
        self.create_simple_object()

        self.obj_collection.update_one(
            {
                "_id": "oid3",
            },
            {
                "$push": {
                    "probabilities": {
                        "classifier_name": "stamp_classifier",
                        "classifier_version": "stamp_classifier_1.0.0",
                        "class_name": "CLASS1",
                        "probability": 1,
                        "ranking": 1,
                    }
                }
            },
        )

        f = self.obj_collection.find_one({"_id": "oid3"})
        expected_object_probabilities = [
            {
                "classifier_name": "stamp_classifier",
                "classifier_version": "stamp_classifier_1.0.0",
                "class_name": "CLASS1",
                "probability": 1,
                "ranking": 1,
            }
        ]
        self.assertEqual(f["probabilities"], expected_object_probabilities)
