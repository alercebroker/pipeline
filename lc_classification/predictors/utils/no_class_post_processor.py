import pandas as pd


class NoClassifiedPostProcessor(object):
    class_name = "no_class"

    def __init__(
        self, messages_dataframe: pd.DataFrame, classifications: pd.DataFrame
    ) -> None:
        self.messages = messages_dataframe
        self.classifications = classifications

    def get_modified_classifications(self) -> pd.DataFrame:
        classifications_columns = self.classifications.columns.values.tolist()
        result_columns_names = classifications_columns + [self.class_name]
        classifications_cloumns_length = len(classifications_columns)
        no_classified_data = ([0] * classifications_cloumns_length) + [1]

        result_data = []
        result_aids = []

        for aid, _ in self.messages.iterrows():
            try:
                aid_classifications = list(self.classifications.loc[aid]) + [0]
            except KeyError:
                aid_classifications = no_classified_data

            result_aids.append(aid)
            result_data.append(aid_classifications)

        result_df = pd.DataFrame(
            result_data, index=result_aids, columns=result_columns_names
        )
        result_df.index.name = "aid"

        return result_df
