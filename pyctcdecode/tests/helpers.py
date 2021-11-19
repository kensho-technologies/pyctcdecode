import os
import shutil
import tempfile
import unittest


class TempfileTestCase(unittest.TestCase):
    def setUp(self):
        """Create a temp dir."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temp dir."""
        shutil.rmtree(self.temp_dir)

    def clear_dir(self):
        """Clear the contents of the temp dir."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir)
