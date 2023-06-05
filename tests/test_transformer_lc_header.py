from alerce_classifiers.transformer_lc_header.model import TranformerLCHeaderClassifier
from alerce_classifiers.transformer_lc_header.mapper import LCHeaderMapper


def test_constructor():
    mapper = LCHeaderMapper()
    model = TranformerLCHeaderClassifier(
        model_path="", header_quantiles_path="", mapper=mapper
    )
    assert model
