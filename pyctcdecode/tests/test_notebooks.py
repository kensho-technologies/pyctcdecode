import os
import unittest

from nbconvert.preprocessors import ExecutePreprocessor  # type: ignore
import nbformat  # type: ignore


CUR_PATH = os.path.dirname(os.path.abspath(__file__))
TUTORIALS_PATH = os.path.join(CUR_PATH, "..", "..", "tutorials")


def run_notebook_from_file(notebook_file):
    """Run an ipython notebook with the right path settings and get the results."""
    with open(notebook_file) as f:
        nb = nbformat.read(f, as_version=4)

    processor = ExecutePreprocessor(timeout=20)
    result = processor.preprocess(nb, {"metadata": {"path": "./tutorials/"}})
    return result


class TestNotebooks(unittest.TestCase):
    def test_notebooks_run(self):
        notebook_file = os.path.join(TUTORIALS_PATH, "00_basic_usage.ipynb")
        result = run_notebook_from_file(notebook_file)
        # we don't really care about the result, just that it came back with no error
        self.assertIsInstance(result, tuple)
        self.assertEqual(list(result[1].keys()), ["metadata"])
