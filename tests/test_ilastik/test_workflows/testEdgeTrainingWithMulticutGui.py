from tests.test_ilastik.helpers import ShellGuiTestCaseBase
import h5py
import logging
import numpy
import sys
import shutil
import tempfile
import time
import threading
import zipfile

from pathlib import Path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from ilastik.applets.dataSelection.opDataSelection import FilesystemDatasetInfo
from ilastik.workflows.edgeTrainingWithMulticut import EdgeTrainingWithMulticutWorkflow
from lazyflow.utility.timer import Timer

from lazyflow.classifiers.parallelVigraRfLazyflowClassifier import ParallelVigraRfLazyflowClassifier

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


def waitProcessEvents(timeout=1, event: threading.Event = None):
    def cond():
        if event:
            return event.is_set()
        else:
            return False

    start = time.time()
    while not cond():
        QApplication.processEvents()
        if time.time() - start > timeout:
            return


class TestEdgeTrainingWithMulticutGui(ShellGuiTestCaseBase):
    """Run a set of GUI-based tests on the multicut workflow.

    Note: These tests are named (prefixed with `test_%02d`) in order to impose
        an order. Tests simulate interaction with a ilastik and depend on
        the earlier ones.
    """

    @classmethod
    def workflowClass(cls):
        return EdgeTrainingWithMulticutWorkflow

    @classmethod
    def setup_class(cls):
        # Base class first
        super().setup_class()

        # input files:
        current_dir = Path(__file__).parent
        cls.sample_data_raw = current_dir.parent / "data" / "inputdata" / "3d.h5"
        cls.sample_data_probs = current_dir.parent / "data" / "inputdata" / "3d_Probabilities.h5"

        # output files:
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.project_file = cls.temp_dir / "test_project_multicut.ilp"
        cls.output_file = cls.temp_dir / "out_multicut_segmentation.h5"

        # reference files

        cls.reference_zip_file = (
            current_dir.parent / "data" / "outputdata" / "testEdgeTrainingWithMulticutReference.zip"
        )
        cls.reference_path = cls.temp_dir / "reference"
        cls.reference_files = {
            "multicut_segmentation_no_rf_t0.6_kerningham-lin": cls.reference_path
            / "multicut_segmentation_no_rf_t0.6_kerningham-lin.h5",
        }
        cls.reference_path.mkdir()
        with zipfile.ZipFile(cls.reference_zip_file, mode="r") as zip_file:
            zip_file.extractall(path=cls.reference_path)
        cls.unzipped_reference_files = [cls.reference_path / fp for fp in zip_file.namelist()]

        for file_name in cls.reference_files.values():
            assert file_name.exists(), file_name

        # Start the timer
        cls.timer = Timer()
        cls.timer.unpause()

    @classmethod
    def teardown_class(cls):
        cls.timer.pause()
        logger.debug(f"Total Time: {cls.timer.seconds()} seconds.")

        # Call our base class so the app quits!
        super().teardown_class()

        # Clean up: Delete any test files we generated
        shutil.rmtree(cls.temp_dir, onerror=lambda *x: logger.error(f"Could not delete file {x}"))

    def test_00_check_preconditions(self):
        """Make sure the needed files exist"""
        needed_files = [self.sample_data_raw]
        for f in needed_files:
            assert f.exists(), f"File {f} does not exist!"

    def test_01_create_project(self):
        """
        Create a blank project, manipulate few couple settings, and save it.
        """

        def impl():
            projFilePath = str(self.project_file)
            shell = self.shell

            # New project
            shell.createAndLoadNewProject(projFilePath, self.workflowClass())
            workflow = shell.projectManager.workflow

            # Add our input files:
            opDataSelection = workflow.dataSelectionApplet.topLevelOperator
            opDataSelection.DatasetGroup.resize(1)
            info_raw = FilesystemDatasetInfo(
                filePath=str(self.sample_data_raw), project_file=self.shell.projectManager.currentProjectFile
            )
            opDataSelection.DatasetGroup[0][EdgeTrainingWithMulticutWorkflow.DATA_ROLE_RAW].setValue(info_raw)

            info_prob = FilesystemDatasetInfo(
                filePath=str(self.sample_data_probs), project_file=self.shell.projectManager.currentProjectFile
            )
            opDataSelection.DatasetGroup[0][EdgeTrainingWithMulticutWorkflow.DATA_ROLE_PROBABILITIES].setValue(
                info_prob
            )
            # Save
            shell.projectManager.saveProject()

        # Run this test from within the shell event loop
        self.exec_in_shell(impl)

    def test_02_do_watershed(self):
        """
        Go to the watershed applet and adjust some settings.
        Apply and check the outcome.
        """

        def impl():
            shell = self.shell
            workflow = shell.projectManager.workflow
            wsdtApplet = workflow.wsdtApplet
            gui = wsdtApplet.getMultiLaneGui().currentGui()
            opWsdt = wsdtApplet.topLevelOperator.getLane(0)

            # activate the preprocessing applet
            shell.setSelectedAppletDrawer(1)
            # let the gui catch up
            QApplication.processEvents()
            self.waitForViews(gui.editor.imageViews)

            # set second channel for probability use
            gui.channel_actions[0].trigger()
            gui.channel_actions[1].trigger()

            # let the gui catch up
            QApplication.processEvents()

            assert opWsdt.ChannelSelections.value == [1]

            threshold = 0.6
            gui.threshold_box.setValue(threshold)
            min_size = 7
            gui.min_size_box.setValue(min_size)
            sigma = 1.1
            gui.sigma_box.setValue(sigma)
            alpha = 0.4
            gui.alpha_box.setValue(alpha)

            # let the gui catch up
            QApplication.processEvents()

            assert opWsdt.Superpixels.ready()
            assert opWsdt.Threshold.value == threshold
            assert opWsdt.MinSize.value == min_size
            assert opWsdt.Sigma.value == sigma
            assert opWsdt.Alpha.value == alpha

            # in order to wait until the preprocessing is finished
            finished = threading.Event()

            def processing_finished():
                nonlocal finished
                finished.set()

            gui.layersUpdated.connect(processing_finished)

            # trigger the preprocessing and wait
            gui.update_ws_button.click()
            waitProcessEvents(timeout=10, event=finished)
            assert finished.is_set()

            superpixels = opWsdt.Superpixels[:].wait()

            assert superpixels.max() == 280
            assert numpy.unique(superpixels).shape == (273,)

            # Save the project
            saveThread = self.shell.onSaveProjectActionTriggered()
            saveThread.join()

        # Run this test from within the shell event loop
        self.exec_in_shell(impl)

    def test_03_multicut_wo_rf(self):
        """
        do multicut on the edge data directly
        """

        def impl():
            shell = self.shell
            workflow = shell.projectManager.workflow
            multicutApplet = workflow.edgeTrainingWithMulticutApplet
            gui = multicutApplet.getMultiLaneGui().currentGui()
            opMulticut = multicutApplet.topLevelOperator.getLane(0)

            # activate the carving applet
            shell.setSelectedAppletDrawer(2)
            # let the gui catch up
            QApplication.processEvents()
            self.waitForViews(gui.editor.imageViews)

            assert not gui.train_edge_clf_box.isChecked()

            threshold = 0.6
            gui.probability_threshold_box.setValue(threshold)
            QApplication.processEvents()

            assert opMulticut.ProbabilityThreshold.value == threshold

            # execute Multicut
            gui.update_button.click()
            QApplication.processEvents()
            # TODO(k-dominik): fix this timeout thing!
            # this is ugly, but for now, don't know how else to let gui and all requests settle
            waitProcessEvents(timeout=0.5)

            # load reference data and compare
            with h5py.File(self.reference_files["multicut_segmentation_no_rf_t0.6_kerningham-lin"], "r") as f:
                mc_segmentation_reference = f["exported_data"][()]
            mc_segmentation = opMulticut.Output[:].wait()
            numpy.testing.assert_array_equal(mc_segmentation, mc_segmentation_reference)

            # Save the project
            saveThread = self.shell.onSaveProjectActionTriggered()
            saveThread.join()

        # Run this test from within the shell event loop
        self.exec_in_shell(impl)

    def test_04_train_rf(self):
        """
        do multicut on the edge data directly
        """

        def impl():
            shell = self.shell
            workflow = shell.projectManager.workflow
            multicutApplet = workflow.edgeTrainingWithMulticutApplet
            gui = multicutApplet.getMultiLaneGui().currentGui()
            opMulticut = multicutApplet.topLevelOperator.getLane(0)

            # activate the carving applet
            shell.setSelectedAppletDrawer(2)
            # let the gui catch up
            QApplication.processEvents()
            self.waitForViews(gui.editor.imageViews)

            assert not gui.train_edge_clf_box.isChecked()
            assert not gui._training_box.isEnabled()

            gui.train_edge_clf_box.setChecked(True)
            QApplication.processEvents()

            assert gui.train_edge_clf_box.isChecked()
            assert gui._training_box.isEnabled()

            # TODO: test for default feature set
            features = {
                "Raw Data": ["standard_sp_mean"],
                "Probabilities-0": [],
                "Probabilities-1": ["standard_edge_mean"],
            }
            opMulticut.FeatureNames.setValue(features)

            # Do our tests at position 0,0,0
            gui.editor.posModel.slicingPos = (27, 30, 25)
            self.waitForViews(gui.editor.imageViews)
            r_coords = [(-10, 0), (15, -28)]
            l_coords = [(19, 6)]

            imgView = gui.editor.imageViews[2]
            for c in r_coords:
                self.clickMouseFromCenter(imgView, c, button=Qt.RightButton)
                QApplication.processEvents()

            for c in l_coords:
                self.clickMouseFromCenter(imgView, c, button=Qt.LeftButton)

            # this is ugly, but for now, don't know how else to let gui and all requests settle
            waitProcessEvents(timeout=0.5)

            labels_expected = {(85, 88): 2, (175, 257): 2, (232, 233): 1}
            labels = opMulticut.EdgeLabelsDict.value
            assert labels == labels_expected

            # check colors (ARGB):
            for c in r_coords:
                assert self.getPixelColor(imgView, c) == 0xFFFF0000

            for c in l_coords:
                assert self.getPixelColor(imgView, c) == 0xFFFFFF00

            assert opMulticut.opClassifierCache.Output.value is None
            assert not gui.live_update_button.isChecked()
            gui.live_update_button.click()

            waitProcessEvents(timeout=0.5)

            gui.live_update_button.click()
            assert not gui.live_update_button.isChecked()

            assert isinstance(opMulticut.opClassifierCache.Output.value, ParallelVigraRfLazyflowClassifier)
            # Save the project
            saveThread = self.shell.onSaveProjectActionTriggered()
            saveThread.join()

        # Run this test from within the shell event loop
        self.exec_in_shell(impl)
