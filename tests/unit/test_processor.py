from prv_candidates_step.core.processor.processor import Processor
from pytest_mock import MockerFixture


def test_compute(mocker: MockerFixture):
    ztf_strat = mocker.patch(
        "prv_candidates_step.core.strategy.ztf_strategy.ZTFPrvCandidatesStrategy"
    )
    ztf_strat.process_prv_candidates.return_value = ([], [])
    processor = Processor(ztf_strat)
    processor.compute({})
    ztf_strat.process_prv_candidates.assert_called_once()


def test_set_strategy(mocker):
    ztf_strat = mocker.patch(
        "prv_candidates_step.core.strategy.ztf_strategy.ZTFPrvCandidatesStrategy"
    )
    atlas_strat = mocker.patch(
        "prv_candidates_step.core.strategy.atlas_strategy.ATLASPrvCandidatesStrategy"
    )
    processor = Processor(ztf_strat)
    processor.strategy = atlas_strat
    assert processor.strategy == atlas_strat
