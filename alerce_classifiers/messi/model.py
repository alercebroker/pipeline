from .mapper import MessiMapper
from .utils import FEATURES_ORDER
from alerce_classifiers.balto.mapper import BaltoMapper
from alerce_classifiers.balto.model import BaltoClassifier
from alerce_classifiers.base.dto import InputDTO
from alerce_classifiers.base.dto import OutputDTO
from alerce_classifiers.base.model import AlerceModel
from joblib import load

import os
import sys
import torch
import validators


class MessiClassifier(AlerceModel):
    def __init__(
        self,
        model_path: str,
        header_quantiles_path: str,
        feature_quantiles_path: str,
        mapper: MessiMapper,
    ):
        super().__init__(model_path, mapper)
        self.local_files = f"/tmp/{type(self).__name__}/features"
        # some ugly hack
        sys.path.append(os.path.join(os.path.dirname(__file__), "../balto"))
        self._header_classifier = BaltoClassifier(
            model_path,
            header_quantiles_path,
            BaltoMapper(),
        )
        self._taxonomy = self._header_classifier._taxonomy
        self._load_feature_quantiles(feature_quantiles_path)

    def _load_feature_quantiles(self, path):
        self.feature_quantiles = {}
        if validators.url(path):
            for feat in FEATURES_ORDER:
                parsed_feat = feat.replace("/", "&&&")
                url = os.path.join(path, f"norm_{parsed_feat}.joblib")
                self.download(url, self.local_files)
            path = self.local_files

        for feat in FEATURES_ORDER:
            parsed_feat = feat.replace("/", "&&&")
            self.feature_quantiles[feat] = load(f"{path}/norm_{parsed_feat}.joblib")

    def _load_model(self, model_path: str) -> None:
        self._local_files = f"/tmp/{type(self).__name__}"
        if validators.url(model_path):
            model_path = self.download(model_path, self._local_files)
        self.model = torch.load(model_path, map_location=torch.device("cpu")).eval()

    def predict(self, data_input: InputDTO) -> OutputDTO:
        input_nn, aid_index = self.mapper.preprocess(
            data_input,
            header_quantiles=self._header_classifier.quantiles,
            feature_quantiles=self.feature_quantiles,
        )

        with torch.no_grad():
            pred = self.model.predict_mix(**input_nn)

        return self.mapper.postprocess(pred, taxonomy=self._taxonomy, index=aid_index)
