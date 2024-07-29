import pandas as pd

from pandas.core.dtypes.common import is_all_strings
from typing import List

from alerce_classifiers.base.dto import OutputDTO
from lc_classification.core.parsers.kafka_parser import (
    KafkaOutput,
    KafkaParser,
)


class ScribeParser(KafkaParser):
    def __init__(self, *, classifier_name: str):
        super().__init__(self)
        self.classifier_name = classifier_name

    def parse(self, to_parse: OutputDTO, **kwargs) -> KafkaOutput[List[dict]]:
        """Parse data output from the Random Forest to scribe commands.
        Parameters
        ----------
        to_parse : OutputDTO
            Output from the model. Has two attributes:
            probabilities and hierarchical.

            OutputDTO.probabilities is a dataframe
            OutputDTO.hierarchical is a Dict[str,pd.DataFrame]
            with two keys: "top" and "children" where children is a Dict[str, pd.DataFrame]

        Examples
        --------
        to_parse.hierarchical

            {
                'top':                     Periodic  Stochastic  Transient
                            oid
                            vbKsodtqMI     0.434        0.21      0.356,
                'children': {
                    'Transient':                    SLSN   SNII   SNIa  SNIbc
                                        oid
                                        vbKsodtqMI  0.082  0.168  0.444  0.306,
                    'Stochastic':                   AGN  Blazar  CV/Nova   QSO    YSO
                                        oid
                                        vbKsodtqMI  0.032   0.056    0.746  0.01  0.156,
                    'Periodic':                     CEP    DSCT   E      LPV    Periodic-Other    RRL
                                        oid
                                        vbKsodtqMI  0.218  0.082  0.158  0.028  0.12            0.394
                }
            }

        to_parse.probabilities
                            SLSN      SNII      SNIa     SNIbc  ...         E       LPV  Periodic-Other       RRL
              oid                                                 ...
              vbKsodtqMI  0.029192  0.059808  0.158064  0.108936  ...  0.068572  0.012152         0.05208  0.170996,
        }
        """
        if len(to_parse.probabilities) == 0:
            return KafkaOutput([])
        probabilities = to_parse.probabilities
        top = pd.DataFrame()
        probabilities["classifier_name"] = self._get_classifier_name()
        top["classifier_name"] = self._get_classifier_name("top")

        results = [top, probabilities]

        results = pd.concat(results)
        if not results.index.name == "oid":
            try:
                results.set_index("oid", inplace=True)
            except KeyError as e:
                if not is_all_strings(results.index.values):
                    raise e

        commands = []

        def get_scribe_messages(classifications_by_classifier: pd.DataFrame):
            class_names = classifications_by_classifier.columns
            for idx, row in classifications_by_classifier.iterrows():
                command = {
                    "collection": "object",
                    "type": "update_probabilities",
                    "criteria": {
                        "_id": idx,
                    },
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

        for oid in results.index.unique():
            results.loc[[oid], :].groupby(
                "classifier_name", group_keys=False
            ).apply(get_scribe_messages)

        return KafkaOutput(commands)

    def _get_classifier_name(self, suffix=None):
        return (
            self.classifier_name
            if suffix is None
            else f"{self.classifier_name}_{suffix}"
        )


class ScoreScribeParser(KafkaParser):
    def __init__(self, *, classifier_name: str):
        super().__init__(self)
        self.detector_name = classifier_name

    def parse(self, to_parse: OutputDTO, **kwargs) -> KafkaOutput[List[dict]]:
        """Parse data output from the Random Forest to scribe commands.
        Parameters
        ----------
        to_parse : OutputDTO
            Output from the model. Has two attributes:
            probabilities and hierarchical.

            OutputDTO.probabilities is a dataframe
            OutputDTO.hierarchical is a Dict[str,pd.DataFrame]
            with two keys: "top" and "children" where children is a Dict[str, pd.DataFrame]

        Examples
        --------
        to_parse.hierarchical

            {
                'top':                     Periodic  Stochastic  Transient
                            oid
                            vbKsodtqMI     0.434        0.21      0.356,
                'children': {
                    'Transient':                    SLSN   SNII   SNIa  SNIbc
                                        oid
                                        vbKsodtqMI  0.082  0.168  0.444  0.306,
                    'Stochastic':                   AGN  Blazar  CV/Nova   QSO    YSO
                                        oid
                                        vbKsodtqMI  0.032   0.056    0.746  0.01  0.156,
                    'Periodic':                     CEP    DSCT   E      LPV    Periodic-Other    RRL
                                        oid
                                        vbKsodtqMI  0.218  0.082  0.158  0.028  0.12            0.394
                }
            }

        to_parse.probabilities
                            SLSN      SNII      SNIa     SNIbc  ...         E       LPV  Periodic-Other       RRL
              oid                                                 ...
              vbKsodtqMI  0.029192  0.059808  0.158064  0.108936  ...  0.068572  0.012152         0.05208  0.170996,
        }
        """
        commands_list = []

        def get_scribe_commands(
            scores_df: pd.Series, detector_name: str, detector_version: str
        ):
            categories = []
            for idx, row in scores_df.items():
                categories.append({"name": idx, "score": row})
            command = {
                "collection": "score",
                "type": "insert",
                "criteria": {
                    "_id": scores_df.name,
                },
                "data": {
                    "detector_name": detector_name,
                    "detector_version": detector_version,
                    "categories": categories,
                },
                "options": {"upsert": True, "set_on_insert": False},
            }

            commands_list.append(command)
            return scores_df

        if len(to_parse.probabilities) == 0:
            return KafkaOutput([])

        detector_version = kwargs["classifier_version"]

        probabilities = to_parse.probabilities
        probabilities.apply(
            get_scribe_commands,
            axis=1,
            **{
                "detector_name": self.detector_name,
                "detector_version": detector_version,
            },
        )

        if (
            to_parse.hierarchical["top"] is not None
            and not to_parse.hierarchical["top"].empty
        ):
            top_probabilities = to_parse.hierarchical["top"]
            top_probabilities.apply(
                get_scribe_commands,
                axis=1,
                **{
                    "detector_name": f"{self.detector_name}_top",
                    "detector_version": detector_version,
                },
            )

        if len(to_parse.hierarchical["children"]) > 0:
            for child_detector in to_parse.hierarchical["children"].keys():
                if not to_parse.hierarchical["children"][child_detector].empty:
                    to_parse.hierarchical["children"][child_detector].apply(
                        get_scribe_commands,
                        axis=1,
                        **{
                            "detector_name": f"{self.detector_name}_{child_detector}",
                            "detector_version": detector_version,
                        },
                    )

        return KafkaOutput(commands_list)
