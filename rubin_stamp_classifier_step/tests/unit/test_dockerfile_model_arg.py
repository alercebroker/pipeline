"""The Dockerfile's `model` build arg must default to rubin in the stage that
runs `poetry install`, so a build without `--build-arg model=...` still
installs a model runtime. A stage-scoped ARG default does not reach a later
stage; only an ARG declared before the first FROM does."""
import os  # build the Dockerfile path
import re  # match the ARG and FROM lines

DOCKERFILE = os.path.join(os.path.dirname(__file__), "..", "..", "Dockerfile")  # step dir / Dockerfile


def instructions():
    """Dockerfile lines that are instructions: no comments, no blanks, stripped."""
    with open(DOCKERFILE) as fh:  # read the file the image is built from
        lines = [line.strip() for line in fh]  # drop indentation and newlines
    return [line for line in lines if line and not line.startswith("#")]  # keep instructions only


def test_model_arg_default_is_declared_before_the_first_from():
    """The default must be global (before any FROM) to reach every stage."""
    first = instructions()[0]  # the very first instruction of the file
    assert re.fullmatch(r"ARG\s+model=rubin", first)  # global default: rubin


def test_builder_stage_redeclares_the_arg_and_installs_with_it():
    """The stage running poetry install must re-declare `ARG model` (bare) so
    the global default applies, and pass it to `--with`."""
    lines = instructions()  # every instruction, in order
    install = next(i for i, line in enumerate(lines) if "poetry install" in line)  # the install line
    stage_start = max(i for i, line in enumerate(lines[:install]) if line.startswith("FROM"))  # its stage's FROM
    stage = lines[stage_start:install]  # instructions of that stage before the install
    assert any(re.fullmatch(r"ARG\s+model", line) for line in stage)  # bare re-declaration present
    assert "--with=${model}" in lines[install]  # the group comes from the arg
