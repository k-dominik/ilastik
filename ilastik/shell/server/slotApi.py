import logging
import typing

import numpy

from lazyflow.slot import Slot, InputSlot, OutputSlot
from lazyflow.operators import OpReorderAxes
from lazyflow.operatorWrapper import OperatorWrapper
from lazyflow.stype import ArrayLike, ValueSlotType
import vigra


logger =  logging.getLogger(__name__)


class WrappedSlot(object):
    def __init__(self, slot: Slot) -> None:
        self._slot = slot


class WrappedValueSlotTypeInputSlot(WrappedSlot):
    def __init__(self, slot: InputSlot) -> None:
        """Summary

        Args:
            slot (Slot): the slot to be wrapped, level 0, or level 1
              this slot will only store values, and not request anything from
              upstream
        """
        print(f'Constructing {type(self)} for {slot}, {slot.stype}')
        assert isinstance(slot, InputSlot)
        if not isinstance(slot.stype, ValueSlotType):
            raise ValueError(
                f'This class only wraps ArrayLike slots. got {slot.stype}.'
            )
        assert slot.level == 0 or slot.level == 1
        super().__init__(slot)
        # need to do any magic before assigning it?
        self.slot = self._slot

    def set_value(self, value: typing.Any, subindex: int=None):
        if self.slot.level == 1:
            if subindex is None:
                raise ValueError("Subindex needs to be given for multi-level-slots!")
            if len(self.slot) >= subindex:
                self.slot.resize(subindex + 1)
            slot = self.slot[subindex]
        else:
            slot = self.slot

        slot.setValue(value)



class WrappedArrayLikeInputSlot(WrappedSlot):
    def __init__(self, slot: InputSlot, incoming_axis_order: str='tczyx') -> None:
        """Summary

        Args:
            slot (Slot): the slot to be wrapped, level 0, or level 1
            incoming_axis_order (str): Axisorder that is used for set_in_slot
        """
        print(f'Constructing {type(self)} for {slot}, {slot.stype}')
        assert isinstance(slot, InputSlot)
        if not isinstance(slot.stype, ArrayLike):
            raise ValueError(
                f'This class only wraps ArrayLike slots. got {slot.stype}.'
            )
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
                subindex: int=None,
                incoming_axis_order: str=None
            ) -> None:
        """Write data into slot

        Args:
            slicing (typing.Tuple[slice]): slicing for each of the axes. Length
              must equal self.incoming_axis_order, or, if supplied,
              incoming_axis_order param of this method
            data (numpy.ndarray): data to be written
            subindex (int): for level1 slots, the sub index.
            incoming_axis_order (str, optional): Overrules
              self.incoming_axis_order
        """
        if self.slot.level == 1:
            if subindex is None:
                raise ValueError("Subindex needs to be given for multi-level-slots!")
            slot = self.slot[subindex]
        else:
            slot = self.slot
        axis_order = incoming_axis_order or self.axis_order
        print(f'axis_order: {axis_order}')
        taggedArray = data.view(vigra.VigraArray)
        taggedArray.axistags = vigra.defaultAxistags(axis_order)
        print(f'tagged_array_axis: {taggedArray.axistags}')

        slot_axis_tags = slot.meta.axistags
        slot_axis_keys = [tag.key for tag in slot_axis_tags]
        print(f'slot_axis_keys: {slot_axis_keys}')
        transposedArray = taggedArray.withAxes(*slot_axis_keys)

        taggedSlicing = dict(list(zip(axis_order, slicing)))
        print(f'taggedSlicing: {taggedSlicing}')
        transposedSlicing = ()
        for k in slot_axis_keys:
            if k in axis_order:
                transposedSlicing += (taggedSlicing[k],)
        print(f'transposedSlicing: {transposedSlicing}')
        slot[transposedSlicing] = transposedArray.view(numpy.ndarray)


class WrappedArrayLikeOutputSlot(WrappedSlot):
    def __init__(self, slot: OutputSlot, forced_axisorder: str='tczxy') -> None:
        assert isinstance(slot, OutputSlot)
        super().__init__(slot)
        assert slot.level == 1, f"Only supporting level 1 slots, got {slot.level}"
        # TODO: implementation for level 1 slots
        if slot.getRealOperator().parent is not None:
            self._op_reorder = OperatorWrapper(
                OpReorderAxes,
                # Here graph, not parent in order to allow wrapping of slots of
                # single operators.
                parent=slot.getRealOperator().parent,
                broadcastingSlotNames=['AxisOrder'])
        else:
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


