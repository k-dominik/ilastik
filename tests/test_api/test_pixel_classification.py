import os
import tempfile

from ilastik import IlastikAPI



class TestPixelClassificationAPI(object):
    """ 
    """
    @classmethod
    def setup_class(cls):
        cls.api = IlastikAPI()
        # temp folder to dump our stuff
        cls.tmp_folder = tempfile.mkdtemp()

    @classmethod
    def teardown_class(cls):
        import shutil
        shutil.rmtree(cls.tmp_folder)

    def test_01_project_creation(self):
        workflow_type = 'Pixel Classification'
        self.project_file = os.path.join(self.tmp_folder, 'pc.ilp')
        self.api.create_project(workflow_type, self.project_file)

        assert os.path.exists(self.project_file)
