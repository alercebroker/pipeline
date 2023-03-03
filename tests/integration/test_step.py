from prv_candidates_step.step import PrvCandidatesStep


def test_step_initialization(kafka_service, env_variables):
    from scripts.run_step import step

    assert isinstance(step, PrvCandidatesStep)
