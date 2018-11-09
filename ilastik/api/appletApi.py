import logging
import typing

from ilastik.api.slotApi import (
    WrappingException,
    WrappedArrayLikeInputSlot, WrappedArrayLikeOutputSlot,
    WrappedValueSlot, WrappedSlot
)
from ilastik.applets.base.applet import Applet
from lazyflow.graph import InputSlot, OutputSlot
from lazyflow.operators import OpReorderAxes
from lazyflow.operatorWrapper import OperatorWrapper
from lazyflow import stype
import collections
from ilastik.applets.dataSelection.dataSelectionApplet import DataSelectionApplet


logger = logging.getLogger(__name__)


class WrappedTLO(object):
    def __init__(
            self,
            applet: Applet,
            input_axis_order: str='tczyx',
            output_axis_order: str='tczyx') -> None:
        """

        # TLOs of applets implement the concept of image lanes
          Therefore one can find
          * level 0 slots (input?!) parameters applying to all lanes
          * level 1 slots (regular case, one slot per image lane)
          * level > 1 slots: will not wrap those atm

        Attributes:
            name: str
            applet_type: typing.Type[Applet]
            input_slots: typing.List[InputSlot]
            output_slots: typing.List[OutputSlot]

        Args:
            applet (Applet): Description
            input_axis_order (str, optional): Description
            output_axis_order (str, optional): Description
        """
        self._applet = None
        self._input_slots = None
        self._output_slots = None
        self._input_axis_order = input_axis_order
        self._output_axis_order = output_axis_order

        self._initialize_applet(applet)

    def _initialize_applet(self, applet: Applet) -> None:
        self._applet = applet
        self.wrap_slots()

    def wrap_slots(self) -> None:
        tlo = self._applet.topLevelOperator
        self._input_slots = collections.OrderedDict()
        self._output_slots = collections.OrderedDict()
        if tlo is None:
            return

        input_slots = collections.OrderedDict()
        for input_slot_name, input_slot in tlo.inputs.items():
            wrapped_slot = None
            try:
                if isinstance(input_slot.stype, stype.ImageLike):
                    wrapped_slot = WrappedArrayLikeInputSlot(
                        input_slot,
                        incoming_axis_order=self._input_axis_order
                    )
                elif isinstance(input_slot.stype, stype.ValueLike):
                    wrapped_slot = WrappedValueSlot(input_slot)
            except WrappingException as e:
                logger.debug(f"Did not wrap input slot {input_slot_name} because of {e}")
                continue

            if wrapped_slot is not None:
                assert input_slot_name not in input_slots
                input_slots[input_slot_name] = {
                    'slot': wrapped_slot
                }

        output_slots = collections.OrderedDict()
        for output_slot_name, output_slot in tlo.outputs.items():
            wrapped_slot = None
            if isinstance(output_slot.stype, stype.ImageLike):
                wrapped_slot = WrappedArrayLikeOutputSlot(
                    output_slot, forced_axisorder=self._output_axis_order)
            elif isinstance(output_slot.stype, stype.ValueLike):
                wrapped_slot = WrappedValueSlot(
                    output_slot)

            if wrapped_slot is not None:
                assert output_slot_name not in output_slots
                output_slots[output_slot_name] = {
                    'slot': wrapped_slot
                }

        self._input_slots = input_slots
        self._output_slots = output_slots

    def get_wrapped_slots(self) -> typing.Dict[str, WrappedSlot]:
        return {
            'input_slots': self._input_slots,
            'output_slots': self._output_slots}

    @property
    def name(self):
        return self._applet.name

    @property
    def applet_type(self):
        return type(self._applet)

    @property
    def input_slots(self):
        return self._input_slots

    @property
    def output_slots(self):
        return self._output_slots


class WrappedApplets(object):
    def __init__(
            self,
            applets: typing.List[Applet],
            input_axis_order: str='tczyx',
            output_axis_order: str='tczyx') -> None:
        self._original_applets: typing.List[Applet] = applets
        self._applets: typing.List[WrappedTLO] = []
        self._input_axis_order = input_axis_order
        self._output_axis_order = output_axis_order
        self._initialize_applets()

    def _initialize_applets(self) -> None:
        for applet in self._original_applets:
            self._initialize_applet(applet)

    def _initialize_applet(self, applet: Applet) -> None:
        # TODO: build and index or something, or memoize getitem
        # TODO: register callbacks
        logger.debug(f'wrapping applet: {applet.name}')
        self._applets.append(WrappedTLO(applet, self._input_axis_order, self._output_axis_order))

    def get_slot_version(self, dataset_name):
        pass

    @property
    def dataset_names(self):
        data_selection_applet = self[DataSelectionApplet]
        opDataSelection = data_selection_applet._applet.topLevelOperator
        dataset_names = list(im.value for im in opDataSelection.ImageName)
        return dataset_names

    def get_states(self, lane_index: int) -> typing.Dict:
        """

        Returns a dict with applet names as keys, and input/output slot dicts
        """
        ret = {}

        for wrapped_TLO in self._applets:
            wrapped_slots = wrapped_TLO.get_wrapped_slots()
            tmp = {}
            for input_slot_name, input_slot in wrapped_slots['input_slots'].items():
                tmp[input_slot_name] = input_slot['slot'].to_dict(subindex=lane_index)

            for output_slot_name, output_slot in wrapped_slots['output_slots'].items():
                tmp[output_slot_name] = output_slot['slot'].to_dict(subindex=lane_index)

            ret[wrapped_TLO.name] = tmp
        return ret

    def get_slot_versions(self, dataset_name):
        # TODO
        lane_index = self.dataset_names.index(dataset_name)
        for applet in self._applets:
            lane_data = applet.get_wrapped_slots(lane_index)
            if lane_data['output_slots'] is None:
                continue
            for output_slot_data in lane_data['output_slots'].values():
                output_slot = output_slot_data['slot']
                if isinstance(output_slot, list):
                    # level 2 slot: skipping
                    continue
                elif output_slot.slot.level == 1:
                    logger.debug(f"{applet.name}, {output_slot.slot.name}, {output_slot.slot.stype}")
                else:
                    continue

    def __getitem__(
            self,
            key: typing.Union[int, str, typing.Type[Applet]]
        ) -> typing.Union[Applet, typing.List[Applet]]:
        """Return applet object

        Args:
            key (int|str|Applet): returns the applet matching the key. Depending
              on the key type, different behaviour is implemented:
              int: simply returns a (single) Applet at index key
              str: returns Applet, or list of Applets with Applet.name matching
                key exactly. List is returns with mutltiple matches
                Applet: returns Applet, or list of Applets
        """
        if isinstance(key, int):
            return self._get_applet_by_index(key)
        elif isinstance(key, str):
            return self._get_applet_by_name(key)
        elif issubclass(key, Applet):
            return self._get_applet_by_type(key)
        else:
            raise ValueError(
                f"Allowed key types are 'int', 'str', and 'Applet'. Got {type(key)} instead")

    def _get_applet_by_index(self, index: int) -> Applet:
        return self._applets[index]

    def _get_applet_by_name(self, name: str) -> typing.Union[Applet, typing.List[Applet]]:
        matches = [applet for applet in self._applets if applet.name == name]
        if len(matches) == 1:
            return matches[0]
        else:
            return matches

    def _get_applet_by_type(self, applet_type: typing.Type[Applet]) -> typing.Union[Applet, typing.List[Applet]]:
        matches = [applet for applet in self._applets if applet.applet_type == applet_type]
        if len(matches) == 1:
            return matches[0]
        else:
            return matches

    def n_lanes(self):
        # checks the data selection applet for number of input images
        data_selection_applet = self[DataSelectionApplet]
        tlo = data_selection_applet._applet.topLevelOperator
        return len(tlo.DatasetGroup)
