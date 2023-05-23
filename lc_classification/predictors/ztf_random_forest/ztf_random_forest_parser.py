import pandas as pd
from lc_classification.predictors.predictor.predictor_parser import (
    Classification,
    PredictorInput,
    PredictorOutput,
    PredictorParser,
)


class ZtfRandomForestPredictorParser(PredictorParser):
    def parse_input(self, to_parse: dict) -> PredictorInput[pd.DataFrame]:
        parsed = PredictorInput(pd.DataFrame())
        return parsed

    def parse_output(self, to_parse: dict) -> PredictorOutput:
        """Parse data output from the Random Forest to a model that the step understands.
        Parameters
        ----------
        to_parse : dict
            a dictionary as returned by the Random Forest with the following data

            .. code-block::

                "hierarchical": {"top": prob_root, "children": resp_children},
                "probabilities": prob_all,
                "class": prob_all.idxmax(axis=1),
        """
        probabilities = to_parse["probabilities"]
        top = to_parse["hierarchical"]["top"]
        children = to_parse["hierarchical"]["children"]

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
        parsed = PredictorOutput([])
        for result in results:
            classification = Classification(result.pop("aid"), result)
            parsed.classifications.append(classification)
        return parsed

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
