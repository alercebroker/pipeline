from typing import List
from lc_classification.core.kafka_parser import KafkaOutput, KafkaParser
import pandas as pd

from lc_classification.predictors.predictor.predictor_parser import PredictorOutput


class ScribeParser(KafkaParser):
    def parse(self, to_parse: PredictorOutput) -> KafkaOutput[List[dict]]:
        """Parse data output from the Random Forest to scribe commands.
        Parameters
        ----------
        to_parse : dict
            a dictionary as returned by the Random Forest with the following data

            .. code-block::

                "hierarchical": {"top": ??, "children": ??},
                "probabilities": pd.DataFrame,
                "class": str,
        """
        probabilities = to_parse.classifications["probabilities"]
        top = to_parse.classifications["hierarchical"]["top"]
        children = to_parse.classifications["hierarchical"]["children"]

        top_ranking = self._get_ranking(top)
        probabilities_ranking = self._get_ranking(probabilities)

        top_result = self._stack_df(top, top_ranking)
        probabilities_result = self._stack_df(probabilities, probabilities_ranking)

        probabilities_result["classifier_name"] = self._get_classifier_name()
        top_result["classifier_name"] = self._get_classifier_name("top")

        results = [top_result, probabilities_result]
        for key in children:
            child_ranking = self._get_ranking(children[key])
            child_result = self._stack_df(children[key], child_ranking)
            child_result["classifier_name"] = self._get_classifier_name(key.lower())
            results.append(child_result)

        results = pd.concat(results)
        results.set_index("aid")
        results = results.to_dict("records")
        commands = []
        for classification in results:
            aid = classification.pop("aid")
            commands.append(
                {
                    "collection": "object",
                    "type": "update_probabilities",
                    "criteria": {"_id": aid},
                    "data": classification,
                    "options": {"upsert": True, "set_on_insert": False},
                }
            )
        return KafkaOutput(commands)

    def _get_ranking(self, df):
        ranking = (-df).rank(axis=1, method="dense", ascending=True).astype(int)
        return ranking

    def _stack_df(self, df, ranking):
        df = df.stack()
        ranking = ranking.stack()
        df.rename("probability", inplace=True)
        ranking.rename("ranking", inplace=True)
        result = pd.concat([df, ranking], axis=1)
        result.index.names = ["aid", "class_name"]
        result.reset_index(inplace=True)
        return result

    def _get_classifier_name(self, suffix=None):
        return "lc_classifier" if suffix is None else f"lc_classifier_{suffix}"
