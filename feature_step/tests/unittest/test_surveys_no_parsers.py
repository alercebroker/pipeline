"""features/utils is removed and nothing imports the old parsers module (spec criterion 2).

The parser, scribe and output functions live under features/surveys. The old
module that re-exported them is deleted, so these tests pin three things: the
old dotted name is not importable, the old directory holds no source file, and
no python file of the package names the old module.

A dev machine that ran the tests before the removal still has
features/utils/__pycache__ on disk. Git does not track it and Python treats a
directory without __init__.py as a namespace package, so the tests accept that
leftover: a cache-only directory is not a module anyone can import code from.

The two search patterns are built by concatenation, so this file does not
contain the text it searches for.
"""
import importlib  # import the old names by their dotted path
import importlib.util  # find_spec, to see what the import system would load
import pathlib  # locate the package directory and walk its files
import sys  # drop cached modules before importing
import unittest  # test case base class
from unittest import mock  # restore sys.modules after each import attempt

FEATURE_STEP = pathlib.Path(__file__).resolve().parents[2]  # feature_step/, two levels above tests/unittest
UTILS_DIR = FEATURE_STEP / "features" / "utils"  # directory the removal deletes
OLD_PACKAGE = "features." + "utils"  # dotted name of the removed package
OLD_MODULE = OLD_PACKAGE + "." + "parsers"  # dotted name of the removed module
OLD_PATTERNS = (  # the two alternatives of the criterion 2 grep, as plain substrings
    "utils" + "." + "parsers",  # dotted path, as in "from <package>.<module> import name"
    "utils import " + "parsers",  # module import, as in "from <package> import <module>"
)  # end of the patterns
WALKED_DIRS = ("features", "tests", "scripts")  # source dirs of the package
SKIPPED_DIRS = (  # directories under WALKED_DIRS that hold no source of the package
    FEATURE_STEP / "scripts" / "local_sim" / "out",  # local simulation output, never committed
)  # end of the skipped dirs


def python_files():
    """Return every .py file under features/, tests/ and scripts/, without caches and the skipped dirs."""
    found = []  # paths kept
    for name in WALKED_DIRS:  # each source dir
        for path in sorted((FEATURE_STEP / name).rglob("*.py")):  # every python file below it, in a stable order
            if "__pycache__" in path.parts:  # compiled caches are not source
                continue  # skip it
            if any(skipped in path.parents for skipped in SKIPPED_DIRS):  # inside a skipped dir
                continue  # skip it
            found.append(path)  # a source file to search
    return found  # all source files


class NoParsersModuleTestCase(unittest.TestCase):
    """features/utils no longer exists as a module, on disk or in any import (criterion 2)."""

    def without_cached_old_modules(self):
        """Hide the old package and its submodules from sys.modules until the test ends."""
        patcher = mock.patch.dict(sys.modules)  # snapshot of sys.modules, restored on stop
        patcher.start()  # from here on, changes to sys.modules are temporary
        self.addCleanup(patcher.stop)  # put the snapshot back after the test
        for name in list(sys.modules):  # every module imported so far
            if name == OLD_PACKAGE or name.startswith(OLD_PACKAGE + "."):  # the old package or one of its modules
                del sys.modules[name]  # a stale entry would make the import succeed without touching the disk
        importlib.invalidate_caches()  # make the finders list the directories again

    def test_old_parsers_module_cannot_be_imported(self):
        """Importing the old parsers module raises ModuleNotFoundError, with or without a leftover cache dir."""
        self.without_cached_old_modules()  # no stale sys.modules entry
        with self.assertRaises(ModuleNotFoundError):  # there is no source file to load
            importlib.import_module(OLD_MODULE)  # the import every old caller used

    def test_old_package_is_not_backed_by_a_file(self):
        """The old package has no spec with a file behind it, and without the directory it cannot be imported at all."""
        self.without_cached_old_modules()  # no stale sys.modules entry
        spec = importlib.util.find_spec(OLD_PACKAGE)  # what the import system would load, None when nothing
        origin = None if spec is None else spec.origin  # file behind the spec; None for a namespace package
        self.assertIsNone(origin)  # no __init__.py: either nothing, or a cache-only leftover directory
        if not UTILS_DIR.exists():  # clean checkout: the directory is gone
            with self.assertRaises(ModuleNotFoundError):  # then the package name does not resolve at all
                importlib.import_module(OLD_PACKAGE)  # the removed package

    def test_utils_directory_has_no_python_file(self):
        """features/utils is absent or holds no .py file at any depth; a leftover __pycache__ alone is accepted."""
        sources = sorted(  # python source files still under the old directory
            str(path.relative_to(FEATURE_STEP)) for path in UTILS_DIR.rglob("*.py")  # empty when the directory is absent
        )  # end of the sources
        self.assertEqual(sources, [])  # none: git tracks nothing there

    def test_no_python_file_names_the_old_module(self):
        """No .py under features/, tests/ or scripts/ contains either pattern of the criterion 2 grep."""
        hits = []  # "path:line" of every offending line
        for path in python_files():  # each source file of the package
            lines = path.read_text(errors="replace").splitlines()  # file content, tolerant of odd bytes
            for number, line in enumerate(lines, start=1):  # each line with its 1-based number
                if any(pattern in line for pattern in OLD_PATTERNS):  # the line names the old module
                    hits.append(f"{path.relative_to(FEATURE_STEP)}:{number}")  # remember where
        self.assertEqual(hits, [])  # none
