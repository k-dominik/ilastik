"""Mediates between ilastikServerShell and ilastikServerAPI"""
from __future__ import print_function, division
import logging
import collections
import copy
import typing

import numpy

from ilastik.applets.dataSelection.opDataSelection import DatasetInfo
from ilastik.shell.server.ilastikServerShell import ServerShell
from ilastik.shell.server.appletApi import WrappedApplet, Applets
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
            'Neural Network Classification'
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

    def add_dataset(self, file_name: str) -> int:
        info = DatasetInfo()
        info.filePath = file_name

        data_selection_applet = self._wrapped_applets[DataSelectionApplet]
        opDataSelection = data_selection_applet._applet.topLevelOperator
        n_lanes = len(opDataSelection.DatasetGroup)
        opDataSelection.DatasetGroup.resize(n_lanes + 1)
        opDataSelection.DatasetGroup[n_lanes][0].setValue(info)

        return n_lanes

    @property
    def applets(self):
        return self._wrapped_applets
    

    def cleanup(self) -> None:
        self._server_shell = None
        self._wrapped_applets = None


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


class IlastikAPI_deprecated(object):
    """Collection of user-friendly methods for interaction with ilastik
    """
    def __init__(self, workflow_type: str=None, project_path: str=None):
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
        super(IlastikAPI, self).__init__()
        self._server_shell: ServerShell = ServerShell()
        self.slot_tracker: SlotTracker = None
        self.available_workflows = list(getAvailableWorkflows())

        # HACK: for now, only support certain workflow types, that have been tested
        # TODO: generalize, write some test for all
        self.allowed_workflows = [
            'Pixel Classification',
            'Neural Network Classification'
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

    def initialize_voxel_server(self):
        multislots = []
        for applet in self.applets:
            print(f'applet: {applet}')
            op = applet.topLevelOperator
            print(f'op: {op}')
            if op is None:
                continue
            # Todo: go through all applets and connect slots to SlotTracker
            tmp_slots = []
            for slotname, slot in op.outputs.items():
                if isinstance(slot.stype, stype.ImageType):
                    print(slotname, slot)
                    tmp_slots.append(slot)
            multislots.extend(tmp_slots)

        data_selection_applet = self.get_data_selection_applet()
        image_name_multislot = data_selection_applet.topLevelOperator.ImageName
        # forcing to neuroglancer axisorder
        self._slot_tracker = SlotTracker(
            image_name_multislot, multislots, forced_axes='tczyx'
        )

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
        from ilastik.workflows.pixelClassification import PixelClassificationWorkflow
        if workflow_type == 'Pixel Classification':
            self._server_shell.createAndLoadNewProject(
                project_path,
                PixelClassificationWorkflow
            )
        else:
            raise ValueError('ProjectType needs to be PixelClassification for now')

    def get_workflow_status(self):
        """Collects information about all applets into a dictionary
        """
        workflow = self._server_shell.workflow
        return workflow

    def get_structured_info(self):
        if self._slot_tracker is None:
            self.initialize_voxel_server()
        dataset_names = self._slot_tracker.get_dataset_names()
        json_states = []
        for lane_number, dataset_name in enumerate(dataset_names):
            states = self._slot_tracker.get_states(dataset_name)
            lane_states = []
            for source_name, state in states.items():
                tmp = collections.OrderedDict(zip(state._fields, state))
                tmp['lane_number'] = lane_number
                tmp['dataset_name'] = dataset_name
                tmp['source_name'] = source_name
                lane_states.append(tmp)
            json_states.append(lane_states)
        return (dataset_names, json_states)

    def get_input_info(self):
        """Gather information about inputs to the current workflow"""
        data_selection_applet = self.get_applet(DataSelectionApplet)
        return data_selection_applet

    # TODO: should be Applets Object
    @property
    def applets(self):
        """Get info on all applets in the current workflow

        Returns:
            list: list of applets of the current workflow
        """
        workflow = self._server_shell.workflow
        applets = None
        if workflow is not None:
            applets = workflow.applets
        return applets

    # TODO: move to Applets object
    def get_applet_by_type(self, applet_type: Applet):
        """
        Args:
            applet_type (Applet or derived): the actual class one is looking
              for.

        Returns:
            Applet
        """
        applets = self.applets
        selected_applet = [applet for applet in applets
                           if isinstance(applet, applet_type)]
        assert len(selected_applet) == 1, (
            "For now, expecting only a single applet of a certain type per workflow. Autocontext?!")
        selected_applet = selected_applet[0]
        return selected_applet

    def get_batch_info(self):
        """Information about what info needs to be supplied for batch processing

        Returns:
            OrderedDict: role - dataset info pairs
        """
        workflow = self._server_shell.workflow
        # should this be the one accessed?
        role_names = workflow.ROLE_NAMES
        batch_data_info = self._get_template_dataset_infos()
        return collections.OrderedDict(
            (role_names[k], v) for k, v in batch_data_info.items())

    # TODO: move to Applets object
    def get_data_selection_applet(self):
        data_selection_applet = self._server_shell.workflow.dataSelectionApplet
        return data_selection_applet

    # TODO: move to Applets object
    def get_batch_applet(self):
        """Get the batch applet from the workflow applets
        """
        applets = self.applets
        batch_processing_applets = [applet for applet in applets
                                    if isinstance(applet, BatchProcessingApplet)]
        assert len(batch_processing_applets) == 1, (
            "Expected only a single batch processing applet per workflow.")
        batch_processing_applet = batch_processing_applets[0]
        return batch_processing_applet

    def process_data(self, data):
        """Process data with the loaded projectk

        TODO: proper check if project is ready to process data.
        TODO: check whether return type might depend on the workflow.

        Args:
            data (ndarray, or list of ndarrays): ndarrays to process with the
              loaded ilastik project.

        Returns:
            ndarray or list of ndarrays:  depending on the input argument,

        """
        is_single = False
        if not isinstance(data, collections.Sequence):
            data = [data]
            is_single = True

        if not all(isinstance(d, numpy.ndarray) for d in data):
            raise ValueError("data has to be numpy.ndarray type")
        inputs = self.get_batch_info()
        raw_info = inputs['Raw Data']
        data_info = [DatasetInfo(preloaded_array=d) for d in data]
        for dinfo in data_info:
            dinfo.axistags = raw_info.axistags
        batch_processing_applet = self.get_batch_applet()

        role_data_dict = collections.OrderedDict({'Raw Input': data_info})
        ret_data = batch_processing_applet.run_export(role_data_dict, export_to_array=True)
        if is_single:
            return ret_data[0]
        else:
            return ret_data

    # TODO: modify as to use the Applets thing
    def add_dataset(self, file_name: str):
        info = DatasetInfo()
        info.filePath = file_name

        data_selection_applet = self.get_applet_by_type(applet_type=DataSelectionApplet)
        opDataSelection = data_selection_applet.topLevelOperator
        n_lanes = len(opDataSelection.DatasetGroup)
        opDataSelection.DatasetGroup.resize(n_lanes + 1)
        opDataSelection.DatasetGroup[n_lanes][0].setValue(info)

    # TODO: should be in Applets?
    def set_value_slot(
            self,
            applet_type: Applet,
            slot_name: str,
            value: typing.Any,
            lane_index: int=0
            ):
        """

        """
        applet = self.get_applet_by_type(applet_type)
        op = applet.topLevelOperator
        if slot_name not in op.inputs:
            raise ValueError(f'No slot found with given name {slot_name}.')
        try:
            lane_view = op.getLane(lane_index)
        except IndexError:
            raise

        lane_view.inputs[slot_name].setValue(value)

    def add_dataset_batch(self, file_name: str):
        """Convenience method to add an image lane with the supplied data

        Args:
            file_name: path to image file to add.

        TODO: proper check if project is ready to process data.
        TODO: this only works for pixel classification

        Returns
            (int) lane index
        """
        is_single = False
        input_axes = None
        if not isinstance(file_name, (list, tuple)):
            data = [file_name]
            is_single = True

        data_selection_applet = self.get_applet(DataSelectionApplet)
        # add a new lane
        opDataSelection = data_selection_applet.topLevelOperator

        # configure roles
        if not is_single:
            raise NotImplementedError("Didn't have time to do that yet...")

        # HACK: just to get it working with pixel classification quickly

        template_infos = self._get_template_dataset_infos(input_axes)
        # Invert dict from [role][batch_index] -> path to a list-of-tuples, indexed by batch_index:
        # [ (role-1-path, role-2-path, ...),
        #   (role-1-path, role-2-path,...) ]
        # datas_by_batch_index = zip( *role_data_dict.values() )

        role_input_datas = list(zip(*collections.OrderedDict({'Raw Input': data}).values()))[0]
        existing_lanes = len(opDataSelection.DatasetGroup)
        opDataSelection.DatasetGroup.resize(existing_lanes + 1)
        lane_index = existing_lanes
        for role_index, data_for_role in enumerate(role_input_datas):
            if not data_for_role:
                continue

            if isinstance(data_for_role, DatasetInfo):
                # Caller provided a pre-configured DatasetInfo instead of a just a path
                info = data_for_role
            else:
                # Copy the template info, but override filepath, etc.
                default_info = DatasetInfo(data_for_role)
                info = copy.copy(template_infos[role_index])
                info.filePath = default_info.filePath
                info.location = default_info.location
                info.nickname = default_info.nickname

            # Apply to the data selection operator
            opDataSelection.DatasetGroup[lane_index][role_index].setValue(info)


    # --------------------------------------------------------------------------
    # NOT SURE YET:
    def get_applet_information(self, applet_index):
        """Generates a dict with applet-information
        """
        workflow = self._server_shell.workflow
        applet = workflow.applets[applet_index]


    # --------------------------------------------------------------------------
    # MIRRORED:

    # From batchprocessing applet
    def _get_template_dataset_infos(self, input_axes=None):
        """
        Sometimes the default settings for an input file are not suitable (e.g. the axistags need to be changed).
        We assume the LAST non-batch input in the workflow has settings that will work for all batch processing inputs.
        Here, we get the DatasetInfo objects from that lane and store them as 'templates' to modify for all batch-processing files.
        """
        template_infos = {}
        data_selection_applet = self.get_applet(DataSelectionApplet)
        # If there isn't an available dataset to use as a template
        if len(data_selection_applet.topLevelOperator.DatasetGroup) == 0:
            num_roles = len(data_selection_applet.topLevelOperator.DatasetRoles.value)
            for role_index in range(num_roles):
                template_infos[role_index] = DatasetInfo()
                if input_axes:
                    template_infos[role_index].axistags = vigra.defaultAxistags(input_axes)
            return template_infos

        # Use the LAST non-batch input file as our 'template' for DatasetInfo settings (e.g. axistags)
        template_lane = len(data_selection_applet.topLevelOperator.DatasetGroup) - 1
        opDataSelectionTemplateView = data_selection_applet.topLevelOperator.getLane(template_lane)

        for role_index, info_slot in enumerate(opDataSelectionTemplateView.DatasetGroup):
            if info_slot.ready():
                template_infos[role_index] = info_slot.value
            else:
                template_infos[role_index] = DatasetInfo()
            if input_axes:
                # Support the --input_axes arg to override input axis order, same as DataSelection applet.
                template_infos[role_index].axistags = vigra.defaultAxistags(input_axes)
        return template_infos
