from typing import List
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from lc_classifier.base import AstroObject


class NotTrainedException(Exception):
    pass


class Classifier(ABC):
    @abstractmethod
    def classify_batch(self, astro_objects: List[AstroObject]) -> None:
        pass

    @abstractmethod
    def classify_single_object(self, astro_object: AstroObject) -> None:
        pass

    @abstractmethod
    def fit(self, astro_objects: List[AstroObject], labels: pd.DataFrame):
        pass

    @abstractmethod
    def save_classifier(self, directory: str):
        pass

    @abstractmethod
    def load_classifier(self, directory: str):
        pass


def all_features_from_astro_objects(astro_objects: List[AstroObject]) -> pd.DataFrame:
    first_object = astro_objects[0]
    features = first_object.features.drop_duplicates(subset=['name', 'fid'])
    features = features.set_index(['name', 'fid'])
    indexes = features.index.values

    feature_list = []
    aids = []
    for astro_object in astro_objects:
        features = astro_object.features.drop_duplicates(subset=['name', 'fid'])
        features = features.set_index(['name', 'fid'])
        feature_list.append(features.loc[indexes]['value'].values)

        metadata = astro_object.metadata
        aid = metadata[metadata['name'] == 'aid']['value'].values[0]
        aids.append(aid)

    df = pd.DataFrame(
        data=np.stack(feature_list, axis=0),
        index=aids,
        columns=['_'.join([str(i) for i in pair]) for pair in indexes]
    )
    return df
