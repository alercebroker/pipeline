import pandas as pd


class NoClassifiedPostProcessor(object):
    class_name = "NotClassified"

    def __init__(
        self, messages_dataframe: pd.DataFrame, classifications: pd.DataFrame
    ) -> None:
        self.messages = messages_dataframe
        self.classifications = classifications

    def get_modified_classifications(self) -> pd.DataFrame:
        if self.messages.index.name != "oid":
            self.messages.set_index("oid", inplace=True)
        classifications_columns = self.classifications.columns.values.tolist()
        result_columns_names = classifications_columns + [self.class_name]
        classifications_cloumns_length = len(classifications_columns)
        no_classified_data = ([0] * classifications_cloumns_length) + [1]

        result_data = []
        result_oids = []

        for oid, _ in self.messages.iterrows():
            try:
                oid_classifications = list(self.classifications.loc[oid]) + [0]
            except KeyError:
                oid_classifications = no_classified_data

            result_oids.append(oid)
            result_data.append(oid_classifications)

        result_df = pd.DataFrame(
            result_data, index=result_oids, columns=result_columns_names
        )
        result_df.index.name = "oid"

        return result_df
