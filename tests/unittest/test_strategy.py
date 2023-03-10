from unittest.mock import MagicMock
from magstats_step.strategies.magstats_computer import MagstatsComputer
from magstats_step.strategies.ztf_strategy import ZTFMagstatsStrategy
from magstats_step.strategies.atlas_strategy import ATLASMagstatsStrategy


def test_context_set_strategy():
    context = MagstatsComputer(ZTFMagstatsStrategy())
    context.strategy = ATLASMagstatsStrategy()
    assert isinstance(context.strategy, ATLASMagstatsStrategy)


def test_context_compute_magtasts():
    strategy = MagicMock(ZTFMagstatsStrategy)
    context = MagstatsComputer(strategy)
    context.compute_magtats([], [])
    strategy.compute_magstats.assert_called_with([], [])


def test_ztf_strategy():
    pass
