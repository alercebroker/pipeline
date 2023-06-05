from typing import List
from lc_classification.core.parsers.kafka_parser import KafkaOutput, KafkaParser
import pandas as pd

from lc_classification.predictors.predictor.predictor_parser import PredictorOutput


class ScribeParser(KafkaParser):
    def parse(self, to_parse: PredictorOutput, **kwargs) -> KafkaOutput[List[dict]]:
        """Parse data output from the Random Forest to scribe commands.
        Parameters
        ----------
        to_parse : PredictorOutput
            Classifications inside PredictorOutput are a dictionary
            as returned by the Random Forest with the following data

            .. code-block::

                "hierarchical": {"top": pd.DataFrame "children": dict}
                "probabilities": pd.DataFrame,
                "class": pd.DataFrame,

        Examples
        --------
        {'hierarchical':
            {
                'top':                     Periodic  Stochastic  Transient
                            aid
                            vbKsodtqMI     0.434        0.21      0.356,
                'children': {
                    'Transient':                    SLSN   SNII   SNIa  SNIbc
                                        aid
                                        vbKsodtqMI  0.082  0.168  0.444  0.306,
                    'Stochastic':                   AGN  Blazar  CV/Nova   QSO    YSO
                                        aid
                                        vbKsodtqMI  0.032   0.056    0.746  0.01  0.156,
                    'Periodic':                     CEP   DSCT      E    LPV  Periodic-Other    RRL
                                        aid
                                        vbKsodtqMI  0.218  0.082  0.158  0.028            0.12  0.394
                }
            },
        'probabilities':                    SLSN      SNII      SNIa     SNIbc  ...         E       LPV  Periodic-Other       RRL
                                    aid                                                 ...
                                    vbKsodtqMI  0.029192  0.059808  0.158064  0.108936  ...  0.068572  0.012152         0.05208  0.170996,
        'class':    aid
                    vbKsodtqMI    RRL
        }
        """
        if len(to_parse.classifications["probabilities"]) == 0:
            return KafkaOutput([])
        probabilities = to_parse.classifications["probabilities"]
        top = to_parse.classifications["hierarchical"]["top"]
        children = to_parse.classifications["hierarchical"]["children"]
        probabilities["classifier_name"] = self._get_classifier_name()
        top["classifier_name"] = self._get_classifier_name("top")

        results = [top, probabilities]
        for key in children:
            children[key]["classifier_name"] = self._get_classifier_name(key.lower())
            results.append(children[key])

        results = pd.concat(results)
        results.set_index("aid", inplace=True)

        commands = []

        # results have:  aid class_name  probability  ranking classifier_name
        def get_scribe_messages(classifications_by_classifier: pd.DataFrame):
            class_names = classifications_by_classifier.columns[:-1]
            for idx, row in classifications_by_classifier.iterrows():
                command = {
                    "collection": "object",
                    "type": "update_probabilities",
                    "criteria": {"_id": idx},
                    "data": {
                        "classifier_name": row["classifier_name"],
                        "classifier_version": kwargs["classifier_version"],
                    },
                    "options": {"upsert": True, "set_on_insert": False},
                }
                for class_name in class_names:
                    command["data"].update({class_name: row[class_name]})
                commands.append(command)
            return classifications_by_classifier

        for aid in results.index.unique():
            results.loc[aid].groupby("classifier_name", group_keys=False).apply(
                get_scribe_messages
            )

        return KafkaOutput(commands)

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
