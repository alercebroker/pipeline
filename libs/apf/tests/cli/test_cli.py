from apf.core.management import new_step
import os
import shutil
import unittest
from click.testing import CliRunner


class CLITest(unittest.TestCase):
    name = "cli_test"

    def test_new_step(self):
        BASE = os.getcwd()
        output_path = os.path.join(BASE, self.name)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        runner = CliRunner()
        result = runner.invoke(new_step, self.name)
        print(result.exc_info)
        self.assertTrue(result.exit_code == 0)
        self.assertTrue(os.path.exists(output_path))

        shutil.rmtree(output_path)
