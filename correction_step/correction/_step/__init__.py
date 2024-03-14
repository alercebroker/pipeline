from .step import CorrectionStep


def run_step():
    CorrectionStep.create_step().start()


def memray_profile():
    import memray

    with memray.Tracker("profile/step_profile.bin"):
        CorrectionStep.create_step().start()
