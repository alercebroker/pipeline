from .kafka_parser import KafkaOutput, KafkaParser
from lc_classification.predictors.predictor.predictor_parser import PredictorOutput
import numpy as np
import pandas as pd


class AlerceParser(KafkaParser):
    def __init__(self):
        super().__init__(None)

    def parse(self, model_output, **kwargs) -> KafkaOutput[list]:
        if len(model_output["probabilities"]) == 0:
            return KafkaOutput([])
        messages = kwargs.get("messages", [])
        features = kwargs.get("features", pd.DataFrame())
        parsed = []
        features.replace({np.nan: None}, inplace=True)
        messages_df = pd.DataFrame(
            [{"aid": message.get("aid")} for message in messages]
        )
        messages_df.drop_duplicates("aid", inplace=True)
        for _, row in messages_df.iterrows():
            aid = row.aid
            try:
                features_aid = features.loc[aid].to_dict()
            except KeyError:
                continue

            tree_aid = self._get_aid_tree(model_output, aid)
            write = {
                "aid": aid,
                "features": features_aid,
                "lc_classification": tree_aid,
            }
            parsed.append(write)

        return KafkaOutput(parsed)

    def _get_aid_tree(self, tree, aid):
        tree_aid = {}
        for key in tree:
            data = tree[key]
            if isinstance(data, pd.DataFrame):
                try:
                    data_cpy = data.set_index("aid")
                    tree_aid[key] = data_cpy.loc[aid].to_dict()
                    if "classifier_name" in tree_aid[key]:
                        tree_aid[key].pop("classifier_name")
                except KeyError as e:
                    if not data.index.name == "aid":
                        raise e
                    else:
                        tree_aid[key] = data.loc[aid].to_dict()
                        if "classifier_name" in tree_aid[key]:
                            tree_aid[key].pop("classifier_name")
            elif isinstance(data, pd.Series):
                tree_aid[key] = data.loc[aid]
            elif isinstance(data, dict):
                tree_aid[key] = self._get_aid_tree(data, aid)
        return tree_aid
