import logging
import typing

import numpy

from lazyflow.slot import Slot, InputSlot, OutputSlot
from lazyflow.operators import OpReorderAxes
from lazyflow.operatorWrapper import OperatorWrapper

import vigra


logger =  logging.getLogger(__name__)


class WrappedSlot(object):
    def __init__(self, slot: Slot) -> None:
        self._slot = slot


class WrappedInputSlot(WrappedSlot):
    def __init__(self, slot: InputSlot, incoming_axis_order: str='tczxy') -> None:
        """Summary

        Args:
            slot (Slot): the slot to be wrapped, level 0, or level 1
            incoming_axis_order (str): Axisorder that is used for set_in_slot
        """
        assert isinstance(slot, InputSlot)
        assert slot.level == 0 or slot.level == 1
        super().__init__(slot)
        # need to do any magic befor assigning it?
        self.slot = self._slot
        self.incoming_axis_order = vigra.defaultAxistags(incoming_axis_order)

    @property
    def axis_order(self):
        return "".join(self.incoming_axis_order.keys())

    def write_into(
                self,
                slicing: typing.Tuple[slice],
                data: numpy.ndarray,
                incoming_axis_order: str=None
            ) -> None:
        """Write data into slot

        Args:
            slicing (typing.Tuple[slice]): slicing for each of the axes. Length
              must equal self.incoming_axis_order, or, if supplied,
              incoming_axis_order param of this method
            data (numpy.ndarray): data to be written
            incoming_axis_order (str, optional): Overrules
              self.incoming_axis_order
        """
        axis_order = incoming_axis_order or self.axis_order
        print(f'axis_order: {axis_order}')
        taggedArray = data.view(vigra.VigraArray)
        taggedArray.axistags = vigra.defaultAxistags(axis_order)

        slot_axis_tags = self._slot.meta.axistags
        slot_axis_keys = [tag.key for tag in slot_axis_tags]
        transposedArray = taggedArray.withAxes(*slot_axis_keys)

        taggedSlicing = dict(list(zip(axis_order, slicing)))
        transposedSlicing = ()
        for k in slot_axis_keys:
            if k in axis_order:
                transposedSlicing += (taggedSlicing[k],)
        self._slot[transposedSlicing] = transposedArray.view(numpy.ndarray)



class WrappedOutputSlot(WrappedSlot):
    def __init__(self, slot: OutputSlot, forced_axisorder: str='tczxy') -> None:
        assert isinstance(slot, OutputSlot)
        super().__init__(slot)
        assert slot.level == 1, f"Only supporting level 1 slots, got {slot.level}"
        self._op_reorder = OperatorWrapper(
            OpReorderAxes,
            # Here graph, not parent in order to allow wrapping of slots of
            # single operators.
            graph=slot.getRealOperator().graph,
            broadcastingSlotNames=['AxisOrder'])
        self._op_reorder.AxisOrder.setValue(forced_axisorder)
        self._op_reorder.Input.connect(slot)
        self.slot = self._op_reorder.Output

    @property
    def axis_order(self):
        return self._op_reorder.AxisOrder.value


