import numpy
import vigra

from lazyflow.operators.opArrayPiper import OpArrayPiper
from lazyflow.graph import Graph, InputSlot, OutputSlot
from lazyflow.operatorWrapper import OperatorWrapper
from ilastik.shell.server.slotApi import WrappedValueSlotTypeInputSlot

from lazyflow.stype import ValueSlotType


class TestwrappedLevelZeroSlotValueLike(object):
    def setup(self):
        self.op_pipe = OpArrayPiper(graph=Graph())
        self.op_pipe.Input.stype = ValueSlotType(self.op_pipe.Input)
        # Sanity checks:
        assert self.op_pipe.Input.level == 0
        assert self.op_pipe.Output.level == 0
        assert type(self.op_pipe.Input.stype) == ValueSlotType

    def test_emtpy_wrapping(self):
        op_pipe = self.op_pipe
        wrapped_input_slot = WrappedValueSlotTypeInputSlot(op_pipe.Input)
        assert wrapped_input_slot.slot == op_pipe.Input
        assert wrapped_input_slot._version == 0

    def test_set_value_setting(self):
        op_pipe = self.op_pipe
        wrapped_input_slot = WrappedValueSlotTypeInputSlot(op_pipe.Input)
        values = [
            'a value',
            1,
        ]
        for index, value in enumerate(values):
            wrapped_input_slot.set_value(value)
            assert op_pipe.Input.value == value
            assert wrapped_input_slot._version == 0, (
                f"encountered {wrapped_input_slot._version}, expected {index + 1}")


class TestwrappedMultiLevelSlotValueLike(object):
    def setup(self):
        self.op_pipe = OperatorWrapper(
            OpArrayPiper,
            graph=Graph()
        )
        self.op_pipe.Input.stype = ValueSlotType(self.op_pipe.Input)
        # Sanity checks:
        assert self.op_pipe.Input.level == 1
        assert self.op_pipe.Output.level == 1
        assert type(self.op_pipe.Input.stype) == ValueSlotType

    def test_emtpy_wrapping(self):
        op_pipe = self.op_pipe
        wrapped_input_slot = WrappedValueSlotTypeInputSlot(op_pipe.Input)
        assert wrapped_input_slot.slot == op_pipe.Input
        assert wrapped_input_slot._version == 0, (
                f"encountered {wrapped_input_slot._version}, expected {index + 1}")

    def test_set_value_setting(self):
        op_pipe = self.op_pipe
        wrapped_input_slot = WrappedValueSlotTypeInputSlot(op_pipe.Input)
        values = [
            'a value',
            1,
        ]
        for index, value in enumerate(values):
            wrapped_input_slot.set_value(value, subindex=0)
            assert op_pipe.Input[0].value == value
            assert wrapped_input_slot._version == index + 1, (
                f"encountered {wrapped_input_slot._version}, expected {index + 1}")

    def test_set_value_setting_auto_resize(self):
        op_pipe = self.op_pipe
        wrapped_input_slot = WrappedValueSlotTypeInputSlot(op_pipe.Input)
        values = [
            'a value',
            1,
        ]
        for index, value in enumerate(values):
            wrapped_input_slot.set_value(value, subindex=index)
            assert op_pipe.Input[index].value == value
            # TODO: properly do this once dirtyness for subslots is implemented
            # right now _version aggregates over subslots
            # at some point one should be able to query multi-level versions?!
            assert wrapped_input_slot._version == index + 1, (
                f"encountered {wrapped_input_slot._version}, expected {index + 1}")
