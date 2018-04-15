import os
import tempfile

from nose.tools import assert_raises  # meh!

import numpy
import h5py

from ilastik import IlastikAPI
from ilastik.applets.dataSelection.dataSelectionApplet import DataSelectionApplet
from ilastik.applets.featureSelection.featureSelectionApplet import FeatureSelectionApplet


def generate_dummy_image(image_file_name, data_shape=[256, 512, 1]):
    data = numpy.zeros(data_shape, dtype=numpy.uint8)
    center = [x // 2 for x in data_shape]
    data[0:center[0], 0:center[1]] = 255
    data[center[0]:, center[1]:] = 255
    with h5py.File(image_file_name) as f:
        g = f.create_group('volume')
        g.create_dataset('raw', data=data)
    return data


class ValueSet(object):
    """Little helper class that acts at a function that saves it's calls
    """

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append({'args': args, 'kwargs': kwargs})


class TestPixelClassificationAPI(object):
    @classmethod
    def setup_class(cls):
        cls.input_axis_order = 'yxc'
        cls.output_axis_order = 'yxc'
        print('creating stuff')
        cls.api = IlastikAPI(input_axis_order=cls.input_axis_order, output_axis_order=cls.output_axis_order)
        print(cls.api)
        # temp folder to dump our stuff
        cls.tmp_folder = tempfile.mkdtemp()
        cls.image_file = os.path.join(cls.tmp_folder, 'image.h5')
        cls.image_data = generate_dummy_image(cls.image_file)

    @classmethod
    def teardown_class(cls):
        import shutil
        shutil.rmtree(cls.tmp_folder)

    def test_01_project_creation(self):
        print(self.api)
        workflow_type = 'Pixel Classification'
        self.project_file = os.path.join(self.tmp_folder, 'pc.ilp')
        self.api.create_project(workflow_type, self.project_file)

        assert os.path.exists(self.project_file)
        assert self.api.applets.n_lanes() == 0

    def test_02_add_data(self):
        self.api.add_dataset(self.image_file)

        data_selection_applet = self.api._wrapped_applets[DataSelectionApplet]
        op_data_selection = data_selection_applet._applet.topLevelOperator

        # sanity check the regular access to the data
        assert len(op_data_selection.DatasetGroup) == 1
        data = op_data_selection.ImageGroup[0][0][:].wait()
        assert data.shape == self.image_data.shape
        numpy.testing.assert_array_equal(data, self.image_data)

        # no check the wrapped output
        wrapped_slots = data_selection_applet.get_lane(0)
        print(wrapped_slots)
        output_data = wrapped_slots['output_slots']['ImageGroup']['slot'][0].slot[0][:].wait()

        assert output_data.shape == self.image_data.shape

    # def test_03_set_value_slot(self):
    #     feature_selection_applet = self.api.get_applet_by_type(FeatureSelectionApplet)
    #     op_feature_selection = feature_selection_applet.topLevelOperator
    #     assert_raises(
    #         ValueError,
    #         self.api.set_value_slot,
    #         FeatureSelectionApplet,
    #         'not_there_for_sure',
    #         None,
    #     )

    #     assert_raises(
    #         IndexError,
    #         self.api.set_value_slot,
    #         FeatureSelectionApplet,
    #         'SelectionMatrix',
    #         None,
    #         9999
    #     )

    #     feature_matrix = numpy.random.randint(0, 2, (6, 7))
    #     feature_matrix[3:, 0] = False

    #     value_set = ValueSet()

    #     op_feature_selection.SelectionMatrix.notifyDirty(value_set)

    #     assert len(value_set.calls) == 0
    #     self.api.set_value_slot(
    #         FeatureSelectionApplet,
    #         'SelectionMatrix',
    #         feature_matrix
    #     )
    #     assert len(value_set.calls) == 1
    #     numpy.testing.assert_array_equal(feature_matrix, op_feature_selection.SelectionMatrix.value)

