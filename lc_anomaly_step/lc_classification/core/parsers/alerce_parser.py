import numpy as np
import pandas as pd

from alerce_classifiers.base.dto import OutputDTO

from .kafka_parser import KafkaOutput, KafkaParser


class AlerceParser(KafkaParser):
    def __init__(self):
        super().__init__(None)

    def parse(
        self, model_output: OutputDTO, *, messages, features, **kwargs
    ) -> KafkaOutput[list]:
        if len(model_output.probabilities) == 0:
            return KafkaOutput([])
        parsed = []
        features.replace({np.nan: None}, inplace=True)
        messages_df = pd.DataFrame(
            [{"oid": message.get("oid")} for message in messages]
        )

        # maybe this won't be enough
        probs_copy = model_output.probabilities.copy()
        probs_copy.pop("classifier_name")
        try:
            probs_copy.set_index("oid", inplace=True)
        except KeyError:
            pass
        tree_output = {
            "probabilities": model_output.probabilities,
            "hierarchical": model_output.hierarchical,
            "class": probs_copy.idxmax(axis=1),
        }

        messages_df.drop_duplicates("oid", inplace=True)
        for _, row in messages_df.iterrows():
            oid = row.oid
            try:
                features_oid = features.loc[oid].to_dict()
            except KeyError:
                continue

            tree_oid = self._get_oid_tree(tree_output, oid)
            write = {
                "oid": oid,
                "features": features_oid,
                "lc_classification": tree_oid,
            }
            parsed.append(write)

        return KafkaOutput(parsed)

    def _get_oid_tree(self, tree, oid):
        tree_oid = {}
        for key in tree:
            data = tree[key]
            if isinstance(data, pd.DataFrame):
                try:
                    data_cpy = data.set_index("oid")
                    tree_oid[key] = data_cpy.loc[oid].to_dict()
                    if "classifier_name" in tree_oid[key]:
                        tree_oid[key].pop("classifier_name")
                except KeyError as e:
                    if not data.index.name == "oid":
                        raise e
                    else:
                        tree_oid[key] = data.loc[oid].to_dict()
                        if "classifier_name" in tree_oid[key]:
                            tree_oid[key].pop("classifier_name")
            elif isinstance(data, pd.Series):
                tree_oid[key] = data.loc[oid]
            elif isinstance(data, dict):
                tree_oid[key] = self._get_oid_tree(data, oid)
        return tree_oid
