import logging
import typing

from ilastik.shell.server.slotApi import (
    WrappedArrayLikeInputSlot, WrappedArrayLikeOutputSlot,
    WrappedValueSlotTypeInputSlot, WrappedSlot

)
from ilastik.applets.base.applet import Applet
from lazyflow.graph import InputSlot, OutputSlot
from lazyflow.operators import OpReorderAxes
from lazyflow.operatorWrapper import OperatorWrapper
from lazyflow import stype
import collections
from ilastik.applets.dataSelection.dataSelectionApplet import DataSelectionApplet


logger = logging.getLogger(__name__)


class WrappedApplet(object):
    def __init__(
            self,
            applet: Applet,
            input_axis_order: str='tczyx',
            output_axis_order: str='tczyx'
            ) -> None:
        """

        # TLOs of applets implement the concept of image lanes
          Therefore one can find
          * level 0 slots (input?!) parameters applying to all lanes
          * level 1 slots (regular case, one slot per image lane)
          * level > 1 slots: will not wrap those atm

          * In pixel classification all output slots are level 1


        Attributes:
          name: str
          applet_type: typing.Type[Applet]
          input_slots: typing.List[InputSlot]
          output_slots: typing.List[OutputSlot]
        """
        self._applet = None
        self._input_slots = None
        self._output_slots = None
        self._input_axis_order = input_axis_order
        self._output_axis_order = output_axis_order

        self._initialize_applet(applet)

    def _initialize_applet(self, applet: Applet) -> None:
        self._applet = applet

    def get_lane(self, lane_index: int) -> typing.Dict[str, WrappedSlot]:
        tlo = self._applet.topLevelOperator
        # TODO: initialize inputs and outputs
        # print(tlo)
        # print(len(tlo.inputs))
        # self._input_slots = collections.OrderedDict()
        # for input_slot_name, input_slot in tlo.inputs.items():
        #     print(input_slot_name)
        #     logger.debug(f"adding input_slot: {input_slot_name}")
        #     # TODO: need to check if connected?!
        #     wrapped_slot = None
        #     if type(input_slot.stype) == stype.ArrayLike:
        #         wrapped_slot = WrappedArrayLikeInputSlot(input_slot)

        #     if wrapped_slot is not None:
        #         self._input_slots[input_slot_name] = {
        #             'slot': wrapped_slot,
        #         }

        # for output_slot in tlo.outputSlots:
        #     logger.debug(f"adding output_slot: {output_slot}")
        lane_view = tlo.getLane(lane_index)
        input_slots = collections.OrderedDict()
        for input_slot_name, input_slot in tlo.inputs.items():
            print(input_slot_name)
            if input_slot.level > 1:
                print(f'Ignoring input slot {input_slot_name}, because level > 1')
                continue
            wrapped_slot = None
            if type(input_slot.stype) == stype.ImageType:
                wrapped_slot = WrappedArrayLikeInputSlot(input_slot)

            if wrapped_slot is not None:
                input_slots[input_slot_name] = {
                    'slot': wrapped_slot
                }

        output_slots = collections.OrderedDict()
        for output_slot_name, output_slot in tlo.outputs.items():
            print(output_slot_name, type(output_slot.stype), output_slot.level)
            if output_slot.level > 2:
                print(f'Ignoring input slot {output_slot_name}, because level > 1')
                continue
            wrapped_slot = None
            if output_slot.level == 2:
                wrapped_slot = []
                for subslot in output_slot:
                    if type(subslot.stype) == stype.ImageType:
                        wrapped_slot.append(WrappedArrayLikeOutputSlot(
                            subslot,
                            forced_axisorder=self._output_axis_order,))
            else:
                if type(output_slot.stype) == stype.ImageType:
                    wrapped_slot = WrappedArrayLikeOutputSlot(
                        output_slot,
                        forced_axisorder=self._output_axis_order
                    )

            if wrapped_slot is not None:
                output_slots[output_slot_name] = {
                    'slot': wrapped_slot
                }

        return {'input_slots': input_slots, 'output_slots': output_slots}


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


class Applets(object):
    def __init__(
            self,
            applets: typing.List[Applet],
            input_axis_order: str='tczyx',
            output_axis_order: str='tczyx') -> None:
        self._applets: typing.List[Applet] = []
        self._input_axis_order = input_axis_order
        self._output_axis_order = output_axis_order
        self._initialize_applets(applets)

    def _initialize_applets(self, applets: typing.List[Applet]) -> None:
        for applet in applets:
            self._initialize_applet(applet)

    def _initialize_applet(self, applet: Applet) -> None:
        # TODO: build and index or something, or memoize getitem
        # TODO: register callbacks
        self._applets.append(WrappedApplet(applet, self._input_axis_order, self._output_axis_order))

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
