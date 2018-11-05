import logging
import typing

import numpy

from lazyflow.slot import Slot, InputSlot, OutputSlot
from lazyflow.operators import OpReorderAxes
from lazyflow.operatorWrapper import OperatorWrapper
from lazyflow.stype import ValueSlotType
import vigra


logger = logging.getLogger(__name__)


class WrappedSlot(object):
    def __init__(self, slot: Slot) -> None:
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
            self._slot.notifyInserted(doMulti)
            self._slot.notifyRemoved(self.set_dirty)

    def set_dirty(self, *args):
        # TODO: handle multi-level slots
        self._version += 1


class WrappedValueSlotTypeInputSlot(WrappedSlot):
    def __init__(self, slot: InputSlot) -> None:
        """Summary

        Args:
            slot (Slot): the slot to be wrapped, level 0, or level 1
              this slot will only store values, and not request anything from
              upstream
        """
        logger.debug(f'Constructing {type(self)} for {slot}, {slot.stype}')
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
