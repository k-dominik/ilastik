import os
import tempfile

import numpy
import h5py

from ilastik import IlastikAPI
from ilastik.applets.dataSelection.dataSelectionApplet import DataSelectionApplet


def generate_dummy_image(image_file_name, size=[256, 512, 1]):
    data = numpy.random.randint(0, 256, size, dtype=numpy.uint8)
    with h5py.File(image_file_name) as f:
        g = f.create_group('volume')
        g.create_dataset('raw', data=data)
    return data


class TestPixelClassificationAPI(object):
    """ 
    """
    @classmethod
    def setup_class(cls):
        cls.api = IlastikAPI()
        # temp folder to dump our stuff
        cls.tmp_folder = tempfile.mkdtemp()
        cls.image_file = os.path.join(cls.tmp_folder, 'image.h5')
        cls.image_data = generate_dummy_image(cls.image_file)

    @classmethod
    def teardown_class(cls):
        import shutil
        shutil.rmtree(cls.tmp_folder)

    def test_01_project_creation(self):
        workflow_type = 'Pixel Classification'
        self.project_file = os.path.join(self.tmp_folder, 'pc.ilp')
        self.api.create_project(workflow_type, self.project_file)

        assert os.path.exists(self.project_file)

    def test_02_add_data(self):
        self.api.add_dataset(self.image_file)

        data_selection_applet = self.api.get_applet_by_type(DataSelectionApplet)
        opDataSelection = data_selection_applet.topLevelOperator

        assert len(opDataSelection.DatasetGroup) == 1
        data = opDataSelection.ImageGroup[0][0][:].wait()
        assert data.shape == self.image_data.shape
        numpy.testing.assert_array_equal(data, self.image_data)

