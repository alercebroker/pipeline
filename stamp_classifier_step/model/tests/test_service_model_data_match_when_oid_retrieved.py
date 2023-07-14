import os
import sys
import unittest

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)

import numpy as np
from deployment.stamp_clf import StampClassifier
from parameters import general_keys


class TestServiceCLFOidMatchWithImage(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        data_path = os.path.join(
            PROJECT_PATH, "../pickles", "training_set_May-06-2020.pkl"
        )
        aux_model = StampClassifier()
        aux_model.data_loader.data_path = data_path
        aux_model.data_loader.set_dumping_data_to_pickle(False)
        self.datasets_dict = aux_model.data_loader.get_preprocessed_datasets_splitted()
        aux_model.data_loader.features_names_list.append("oid")
        self.datasets_dict_oid_andfeatures_and_oids = (
            aux_model.data_loader.get_preprocessed_datasets_splitted()
        )

    def _test_oids_retrieved(self, set_name):
        dataset = self.datasets_dict[set_name]
        dataset_oid_andfeatures_and_oids = self.datasets_dict_oid_andfeatures_and_oids[
            set_name
        ]
        n_features_dataset = dataset.meta_data.shape[-1]
        n_features_dataset_oid_andfeatures_and_oids = (
            dataset_oid_andfeatures_and_oids.meta_data.shape[-1]
        )
        self.assertEqual(
            n_features_dataset, n_features_dataset_oid_andfeatures_and_oids - 1
        )
        self.assertEqual(type(dataset_oid_andfeatures_and_oids.meta_data[0, -1]), str)

    def test_oids_retrieved_train(self):
        self._test_oids_retrieved(general_keys.TRAIN)

    def test_oids_retrieved_val(self):
        self._test_oids_retrieved(general_keys.VALIDATION)

    def test_oids_retrieved_test(self):
        self._test_oids_retrieved(general_keys.TEST)

    def _test_images_match_when_including_oid_as_metadata(self, dataset_name):
        dataset = self.datasets_dict[dataset_name]
        dataset_oid_andfeatures = self.datasets_dict_oid_andfeatures_and_oids[
            dataset_name
        ]
        oids = dataset_oid_andfeatures.meta_data[:, -1]
        for oid_idx in range(len(oids)):
            agreement_i = np.mean(
                dataset.data_array[oid_idx]
                == dataset_oid_andfeatures.data_array[oid_idx]
            )
            self.assertEqual(
                agreement_i,
                1.0,
                msg="oid %s, agreement %s" % (oids[oid_idx], str(agreement_i)),
            )

    def test_images_match_when_including_oid_as_metadata_train(self):
        self._test_images_match_when_including_oid_as_metadata(general_keys.TRAIN)

    def test_images_match_when_including_oid_as_metadata_val(self):
        self._test_images_match_when_including_oid_as_metadata(general_keys.VALIDATION)

    def test_images_match_when_including_oid_as_metadata_test(self):
        self._test_images_match_when_including_oid_as_metadata(general_keys.TEST)

    def _test_features_match_when_including_oid_as_metadata(self, dataset_name):
        dataset = self.datasets_dict[dataset_name]
        dataset_oid_andfeatures = self.datasets_dict_oid_andfeatures_and_oids[
            dataset_name
        ]
        oids = dataset_oid_andfeatures.meta_data[:, -1]
        for oid_idx in range(len(oids)):
            agreement_i = np.mean(
                dataset.meta_data[oid_idx]
                == dataset_oid_andfeatures.meta_data[oid_idx][:-1]
            )
            self.assertEqual(
                agreement_i,
                1.0,
                msg="oid %s, agreement %s" % (oids[oid_idx], str(agreement_i)),
            )

    def test_features_match_when_including_oid_as_metadata_train(self):
        self._test_features_match_when_including_oid_as_metadata(general_keys.TRAIN)

    def test_features_match_when_including_oid_as_metadata_val(self):
        self._test_features_match_when_including_oid_as_metadata(
            general_keys.VALIDATION
        )

    def test_features_match_when_including_oid_as_metadata_test(self):
        self._test_features_match_when_including_oid_as_metadata(general_keys.TEST)


if __name__ == "__main__":
    unittest.main()
