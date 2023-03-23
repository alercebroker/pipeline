from lightcurve_step.step import LightcurveStep


def test_step_initialization(kafka_service, env_variables):
    from scripts.run_step import step_creator

    assert isinstance(step_creator(), LightcurveStep)


def test_step_start(kafka_service, env_variables):
    from scripts.run_step import step_creator

    step_creator().start()
