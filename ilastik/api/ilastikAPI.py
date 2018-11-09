"""Mediates between ilastikServerShell and ilastikServerAPI"""
from __future__ import print_function, division
import logging
import collections
import copy
import typing

import numpy

from ilastik.applets.dataSelection.opDataSelection import DatasetInfo
from ilastik.shell.server.ilastikServerShell import ServerShell
from ilastik.api.appletApi import WrappedApplet, Applets
from ilastik.applets.batchProcessing.batchProcessingApplet import BatchProcessingApplet
from ilastik.applets.dataSelection.dataSelectionApplet import DataSelectionApplet
from ilastik.applets.dataSelection.opDataSelection import DatasetInfo
from ilastik.applets.pixelClassification import PixelClassificationApplet
from ilastik.applets.base.applet import Applet
from ilastik.workflow import getAvailableWorkflows

from lazyflow import stype

from ilastik.workflow import Workflow

logger = logging.getLogger(__name__)


class _IlastikAPI(object):
# TODO: add MemoryMonitor

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(
            self,
            workflow_type: str=None,
            project_path: str=None,
            input_axis_order: str='tczyx',
            output_axis_order: str='tczyx'):
        """Initialize a new API object

        If `workflow_type` is given and `project_path` is None, an in-memory
        project is created.
        If `project_path` is given and `workflow_type` is not given, an existing
        project is assumed.
        If both `workflow_type` and `project_path` are given, a new project with
        the given project path is created on disc.

        Args:
            workflow_type (str): workflow type as string (display name),
              e.g. 'Pixel Classification'. Valid workflow types:

              * Pixel Classification
              * Neural Network Classification
              * ...

            project_path (str): path to project
        """
        super().__init__()
        self._server_shell: ServerShell = None
        self._wrapped_applets: typing.Dict[str, WrappedApplet] = None
        self.available_workflows: typing.list[typing.Tuple[Workfklow, str, str]] = \
            list(getAvailableWorkflows())
        self._input_axis_order = input_axis_order
        self._output_axis_order = output_axis_order

        # HACK: for now, only support certain workflow types, that have been tested
        # TODO: generalize, write some test for all
        self.allowed_workflows = [
            'Pixel Classification',
            # 'Neural Network Classification'
        ]

        if workflow_type is not None and project_path is not None:
            # create new project
            self.create_project(workflow_type, project_path)
        elif workflow_type is not None:
            # create in-memory project
            raise NotImplementedError(
                'Creation of in-memory projects not yet supported.'
                'Please supply a file-name'
            )
        elif project_path is not None:
            # load project
            self.load_project_file(project_path)

    def create_project(self, workflow_type: str='Pixel Classification', project_path: str=None):
        """Create a new project

        TODO: memory-only project

        Args:
            project_path (str): path to project file, will be overwritten
              without warning.
            workflow_type (str): workflow type as string,
            using the display name
              e.g. `Pixel Classification`. Valid workflow types:

              * Pixel Classification
              * ...

        Raises:
            ValueError: if an unsupported `workflow_type` is given
        """
        self.cleanup()
        # get display names:
        workflow_names = {x[2]: x[0] for x in self.available_workflows}
        if workflow_type not in workflow_names:
            raise NotImplementedError(
                f'Workflow {workflow_type} can not be found. '
                'Please make sure to use the proper display name.'
            )

        # TODO: remove, once all workflows are supported
        if workflow_type not in self.allowed_workflows:
            raise NotImplementedError(
                f'Workflow {workflow_type} has not been tested yet.'
            )

        if project_path is None:
            raise NotImplementedError('memory-only projects have to be implemented')

        self._server_shell = ServerShell()

        if workflow_type == 'Pixel Classification':
            self._server_shell.createAndLoadNewProject(
                project_path,
                workflow_names[workflow_type]
            )
        else:
            raise ValueError('ProjectType needs to be PixelClassification for now')

        self.initialize_wrappers()

    def load_project_file(self, project_file_path: str):
        """Load project file from disk (local)

        Args:
          project_file_path (str): path of `.ilp` file
        """
        self.cleanup()
        self._server_shell = ServerShell()
        self._server_shell.openProjectFile(project_file_path)
        self.initialize_wrappers()
        pc_applet = self._wrapped_applets[PixelClassificationApplet]
        tlo = pc_applet._applet.topLevelOperator
        tlo.FreezePredictions.setValue(True)
        tlo.FreezePredictions.setValue(False)

    def initialize_wrappers(self):
        self._wrapped_applets = collections.OrderedDict()
        if self._server_shell is None:
            return

        applets = self._server_shell.applets
        self._wrapped_applets = Applets(applets, self._input_axis_order, self._output_axis_order)
        self.initialize_voxel_server()

    def add_dataset(self, file_name: str) -> int:
        info = DatasetInfo()
        info.filePath = file_name

        data_selection_applet = self._wrapped_applets[DataSelectionApplet]
        opDataSelection = data_selection_applet._applet.topLevelOperator
        n_lanes = len(opDataSelection.DatasetGroup)
        opDataSelection.DatasetGroup.resize(n_lanes + 1)
        opDataSelection.DatasetGroup[n_lanes][0].setValue(info)

        # self.initialize_voxel_server()
        return n_lanes

    @property
    def applets(self):
        return self._wrapped_applets

    @property
    def dataset_names(self):
        return self._wrapped_applets.dataset_names

    def cleanup(self) -> None:
        self._server_shell = None
        self._wrapped_applets = None

    def get_structured_info(self):
        pass

class IlastikAPI(_IlastikAPI):
    """
    Main API class Singleton

    right now we enforce only one API instance per interpreter
    Hence, only one workflow can be active at a given time to be on the safe
    side.

    """
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls, *args, *kwargs)

        return cls.__instance
