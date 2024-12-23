from alerce_classifiers.mbappe.mapper import MbappeMapper
from alerce_classifiers.mbappe.model import MbappeClassifier

import pickle

model_path = 'tmp/MbappeClassifier/my_best_checkpoint-step=58684.ckpt'
metadata_quantiles_path = 'tmp/MbappeClassifier'
features_quantiles_path = 'tmp/MbappeClassifier'

model = MbappeClassifier(
    model_path=model_path,
    metadata_quantiles_path=metadata_quantiles_path,
    features_quantiles_path=features_quantiles_path,
    mapper=MbappeMapper(),
)

path_input_dto = 'tests/input_dto/input_dto_ztf_forced.pkl'

with open(path_input_dto, 'rb') as archivo:
    input_dto = pickle.load(archivo)

df_probs = model.predict(input_dto)