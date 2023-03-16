import logging

try:
    from correction_step import CorrectionStep
except ImportError:
    import os
    import sys

    SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
    PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
    sys.path.append(PACKAGE_PATH)
    from correction_step import CorrectionStep


def step_creator():
    from settings import settings_creator

    settings = settings_creator()
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if settings["LOGGING_DEBUG"]:
        level = logging.DEBUG
    return CorrectionStep(config=settings, level=level)


if __name__ == "__main__":
    step_creator().start()
