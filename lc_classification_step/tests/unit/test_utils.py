import unittest
import pandas as pd
from tests.mockdata.mock_data_for_utils import (
    messages_df,
    complete_classifications_df,
    incomplete_classifications_df,
)
from lc_classification.core.utils.no_class_post_processor import (
    NoClassifiedPostProcessor,
)


class NoClassifiedPostProcessorTestCase(unittest.TestCase):
    def test_all_aid_classified(self):
        expected_df = pd.DataFrame(
            [
                [0.1, 0.2, 0.7, 0],
                [0.3, 0.1, 0.6, 0],
                [0.8, 0.1, 0.1, 0],
                [0.2, 0.5, 0.3, 0],
                [0.6, 0.2, 0.2, 0],
            ],
            index=[
                "aid1",
                "aid2",
                "aid3",
                "aid4",
                "aid5",
            ],
            columns=["class1", "class2", "class3", "NotClassified"],
        )
        expected_df.index.name = "aid"

        procesor = NoClassifiedPostProcessor(
            messages_df, complete_classifications_df
        )
        result_df = procesor.get_modified_classifications()

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_some_aid_not_classified(self):
        expected_df = pd.DataFrame(
            [
                [0.1, 0.2, 0.7, 0],
                [0, 0, 0, 1],
                [0.8, 0.1, 0.1, 0],
                [0, 0, 0, 1],
                [0.6, 0.2, 0.2, 0],
            ],
            index=[
                "aid1",
                "aid2",
                "aid3",
                "aid4",
                "aid5",
            ],
            columns=["class1", "class2", "class3", "NotClassified"],
        )
        expected_df.index.name = "aid"

        procesor = NoClassifiedPostProcessor(
            messages_df, incomplete_classifications_df
        )
        result_df = procesor.get_modified_classifications()

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_unordered_data(self):
        pass
