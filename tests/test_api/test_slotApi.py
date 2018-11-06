import numpy
import vigra

from lazyflow.operators.opArrayPiper import OpArrayPiper
from lazyflow.graph import Graph, InputSlot, OutputSlot
from lazyflow.operatorWrapper import OperatorWrapper
from ilastik.shell.server.slotApi import (
    WrappedArrayLikeInputSlot, WrappedArrayLikeOutputSlot,
    WrappedValueSlotTypeInputSlot
)
from lazyflow.stype import ArrayLike, ValueSlotType


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


class TestWrappedMultilevelSlotArrayLike(object):
    def setup(self):
        self.op_pipe = OperatorWrapper(
            OpArrayPiper,
            graph=Graph()
        )

        self.op_pipe.Input.stype = ArrayLike(self.op_pipe.Input)

        # Sanity checks
        assert type(self.op_pipe.Input.stype) == ArrayLike
        assert self.op_pipe.Input.level == 1
        assert self.op_pipe.Output.level == 1
        assert len(self.op_pipe.Input) == 0
        assert len(self.op_pipe.Input) == 0

    def test_slot_wrapping_empty(self):
        op_pipe = self.op_pipe

        default_axisorder = 'tczyx'

        wrapped_input_slot = WrappedArrayLikeInputSlot(
            op_pipe.Input, incoming_axis_order=default_axisorder)
        wrapped_output_slot = WrappedArrayLikeOutputSlot(
            op_pipe.Output, forced_axisorder=default_axisorder)

        assert wrapped_output_slot.axis_order == 'tczyx'
        assert wrapped_input_slot.axis_order == 'tczyx'

    def test_slot_multilevel_wrapping_w_content(self):
        op_pipe = self.op_pipe
        data = numpy.random.randint(
            0, 255, (5, 2, 10, 40, 70), dtype=numpy.uint8)

        default_axisorder = 'tczyx'
        op_pipe.Input.resize(2)
        assert len(op_pipe.Input) == 2
        assert len(op_pipe.Input) == 2

        op_pipe.Input[0].meta.axistags = vigra.defaultAxistags(default_axisorder)
        op_pipe.Input[0].setValue(numpy.zeros((5, 2, 10, 40, 70), dtype=numpy.uint8))

        wrapped_input_slot = WrappedArrayLikeInputSlot(
            op_pipe.Input, incoming_axis_order=default_axisorder)
        wrapped_output_slot = WrappedArrayLikeOutputSlot(
            op_pipe.Output, forced_axisorder=default_axisorder)

        assert wrapped_input_slot.slot.level == 1
        assert wrapped_output_slot.slot.level == 1
        assert wrapped_output_slot.axis_order == default_axisorder
        assert wrapped_input_slot.axis_order == default_axisorder

        slicing = (
            slice(0, 1),    # t
            slice(0, 2),    # c
            slice(0, 10),   # z
            slice(0, 20),   # y
            slice(0, 30)    # x
        )

        data_slice = data[slicing]

        wrapped_input_slot.write_into(
            slicing=slicing,
            data=data_slice,
            subindex=0
        )

        output = wrapped_input_slot.slot[0][:].wait()
        expected_output = numpy.zeros((5, 2, 10, 40, 70), dtype=numpy.uint8)
        expected_output[slicing] = data_slice
        numpy.testing.assert_array_equal(output, expected_output)

        # get output from wrapped slot
        wrapped_output = wrapped_output_slot.get(slicing=slicing, subindex=0)
        numpy.testing.assert_array_equal(wrapped_output, data_slice)
