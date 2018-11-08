import logging
import typing

from ilastik.api.slotApi import (
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

    def wrap_slots(self):
        tlo = self._applet.topLevelOperator
        if tlo is None:
            return {'input_slots': None, 'output_slots': None}


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
