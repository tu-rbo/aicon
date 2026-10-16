"""Template for new unittest-style tests: fixtures in setUp/tearDown, printed output captured."""
# pylint: disable=missing-class-docstring, missing-function-docstring

import io
import sys
import unittest


class Test(unittest.TestCase):
    def setUp(self):
        # Create shared variable
        self.shared_var = 0
        self.str_output = ""
        # redirect stdout to capture print output
        self.captured_output = io.StringIO()
        sys.stdout = self.captured_output

    def tearDown(self):
        # reset redirect
        sys.stdout = sys.__stdout__
        self.str_output = ""

    def test_function(self):
        pass


if __name__ == "__main__":
    unittest.main()
