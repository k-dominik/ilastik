###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2018, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Lesser GNU General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# See the files LICENSE.lgpl2 and LICENSE.lgpl3 for full text of the
# GNU Lesser General Public License version 2.1 and 3 respectively.
# This information is also available on the ilastik web site at:
#          http://ilastik.org/license/
###############################################################################
import logging
import typing

import numpy

from lazyflow.slot import Slot, InputSlot, OutputSlot
from lazyflow.operators import OpReorderAxes
from lazyflow.operatorWrapper import OperatorWrapper
from lazyflow.stype import ArrayLike, ValueLike
import vigra


logger = logging.getLogger(__name__)


class WrappingException(Exception):
    pass


class WrappedSlot(object):
    def __init__(self, slot: Slot) -> None:
        if isinstance(slot, InputSlot):
            if slot.upstream_slot is not None:
                raise WrappingException(
                    f"slot {slot.name} is already connected - connected slots are not wrapped.")

        self._slot = slot
        self._version = 0

        # keep track of dirtyness/versions
        # TODO: properly handle multi-level slots
        def doMulti(slot: Slot, index: int, size):
            slot[index].notifyDirty(self.set_dirty)
            slot[index].notifyValueChanged(self.set_dirty)

        if self._slot.level == 0:
            self._slot.notifyDirty(self.set_dirty)
            self._slot.notifyValueChanged(self.set_dirty)
        else:
            # initialize slots that are already there:
            for subslot in self._slot:
                assert subslot.level == 0
                subslot.notifyDirty(self.set_dirty)
                subslot.notifyValueChanged(self.set_dirty)
            self._slot.notifyInserted(doMulti)
            self._slot.notifyRemoved(self.set_dirty)

    def set_dirty(self, *args):
        # TODO: handle multi-level slots
        logger.debug(f"incrementing version of {self._slot.name}")
        self._version += 1

    @property
    def name(self):
        return self._slot.name

    def to_dict(self):
        raise NotImplementedError


class WrappedValueSlot(WrappedSlot):
    def __init__(self, slot: Slot) -> None:
        """
        Depending on the slot (whether it is an input or output) method getting/
        setting is different:
          * input slots: read/write: necessary to synchronize with the client
          * output slots: read-only

        Args:
            slot (Slot): the slot to be wrapped, leve):
              this slot will only store values, and not request anything from
              upstream
        """
        logger.debug(f'Constructing {type(self)} for {slot}, {slot.stype}')
        if not isinstance(slot.stype, ValueLike):
            raise ValueError(
                f'This class only wraps ArrayLike slots. got {slot.stype}.'
            )
        assert slot.level == 0 or slot.level == 1

        super().__init__(slot)
        # need to do any magic before assigning it?
        self.slot = self._slot

    def set_value(self, value: typing.Any, subindex: int=None):
        assert isinstance(self.slot, InputSlot), "Only input slots support value setting!"
        if self.slot.level == 1:
            if subindex is None:
                raise ValueError("Subindex needs to be given for multi-level-slots!")
            if len(self.slot) >= subindex:
                self.slot.resize(subindex + 1)
            slot = self.slot[subindex]
        else:
            slot = self.slot

        slot.setValue((value))

    def get_value(self, subindex: int=None):
        if self.slot.level == 1:
            if subindex is None:
                raise ValueError("Subindex needs to be given for multi-level-slots!")
            if len(self.slot) >= subindex:
                self.slot.resize(subindex + 1)
            slot = self.slot[subindex]
        else:
            slot = self.slot

        return slot.value

    def to_dict(self, subindex: int=None):
        if self.slot.level == 1:
            if subindex is None:
                raise ValueError("Subindex needs to be given for multi-level-slots!")
            slot = self.slot[subindex]
        else:
            slot = self.slot

        is_input = isinstance(self.slot, InputSlot)
        slot_type = (is_input and 'value input') or 'value output'
        if slot.ready():
            return {
                'slot_type': slot_type,
                'name': slot.name,
                'dtype': repr(type(slot.value)),
                'version': self._version,
                'ready': slot.ready()
            }
        else:
            return {
                'slot_type': slot_type,
                'name': slot.name,
                'version': self._version,
                'ready': slot.ready()
            }


class WrappedArrayLikeInputSlot(WrappedSlot):
    def __init__(self, slot: InputSlot, incoming_axis_order: str='tczyx') -> None:
        """Summary

        Args:
            slot (Slot): the slot to be wrapped, level 0, or level 1
            incoming_axis_order (str): Axisorder that is used for set_in_slot
        """
        logger.debug(f'Constructing {type(self)} for {slot}, {slot.stype}')
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
            incoming_axis_order: str=None) -> None:
        """Write data into slot

        Super special case for scribble annotations, should not be used somewhere
        else.

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
        logger.debug(f'axis_order: {axis_order}')
        taggedArray = data.view(vigra.VigraArray)
        taggedArray.axistags = vigra.defaultAxistags(axis_order)
        logger.debug(f'tagged_array_axis: {taggedArray.axistags}')

        slot_axis_tags = slot.meta.axistags
        slot_axis_keys = [tag.key for tag in slot_axis_tags]
        logger.debug(f'slot_axis_keys: {slot_axis_keys}')
        transposedArray = taggedArray.withAxes(*slot_axis_keys)

        taggedSlicing = dict(list(zip(axis_order, slicing)))
        logger.debug(f'taggedSlicing: {taggedSlicing}')
        transposedSlicing = ()
        for k in slot_axis_keys:
            if k in axis_order:
                transposedSlicing += (taggedSlicing[k],)
        logger.debug(f'transposedSlicing: {transposedSlicing}')
        write_view = transposedArray.view(numpy.ndarray)
        slot[transposedSlicing] = write_view

    def to_dict(self, subindex: int=None):
        if self.slot.level == 1:
            if subindex is None:
                raise ValueError("Subindex needs to be given for multi-level-slots!")
            slot = self.slot[subindex]
        else:
            slot = self.slot
        if slot.ready():
            return {
                'slot_type': 'input',
                'name': slot.name,
                'axes': self.axis_order,
                'dtype': slot.meta.dtype.__name__,
                'shape': slot.meta.shape,
                'version': self._version,
                'ready': slot.ready()
            }
        else:
            return {
                'slot_type': 'input',
                'name': slot.name,
                'axes': self.axis_order,
                'version': self._version,
                'ready': slot.ready()
            }


class WrappedArrayLikeOutputSlot(WrappedSlot):
    def __init__(self, slot: OutputSlot, forced_axisorder: str='tczyx') -> None:
        """Summary

        Args:
            slot (OutputSlot): Description
            forced_axisorder (str, optional): Description
        """
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

    def get(
            self,
            slicing: typing.Tuple[slice],
            subindex: int=None,
            outgoing_axis_order: str=None) -> numpy.ndarray:
        """Summary

        Args:
            slicing (Typing.Tuple[slice]): Description
            subindex (int, optional): Description
            outgoing_axis_order (str, optional): Description, for now not supported

        # TODO: implement outgoing axis order

        Raises:
            NotImplementedError: Description
            ValueError: Description
        """
        if self.slot.level == 1:
            if subindex is None:
                raise ValueError("Subindex needs to be given for multi-level-slots!")
            slot = self.slot[subindex]
        else:
            # slot = self.slot
            raise NotImplementedError

        if outgoing_axis_order is not None:
            if outgoing_axis_order != self.axis_order:
                # TODO, some quick reordering!
                raise NotImplementedError

        return slot[slicing].wait()

    def to_dict(self, subindex: int=None):
        if self.slot.level == 1:
            if subindex is None:
                raise ValueError("Subindex needs to be given for multi-level-slots!")
            slot = self.slot[subindex]
        else:
            slot = self.slot
        if slot.ready():
            return {
                'slot_type': 'output',
                'name': slot.name,
                'axes': self.axis_order,
                'dtype': slot.meta.dtype.__name__,
                'shape': slot.meta.shape,
                'version': self._version,
                'ready': slot.ready()
            }
        else:
            return {
                'slot_type': 'output',
                'name': slot.name,
                'axes': self.axis_order,
                'version': self._version,
                'ready': slot.ready()
            }
