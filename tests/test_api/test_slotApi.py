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
import itertools
import numpy
import vigra

from lazyflow.operators.opArrayPiper import OpArrayPiper
from lazyflow.operators.opValuePiper import OpValuePiper
from lazyflow.graph import Graph, InputSlot, OutputSlot
from lazyflow.operatorWrapper import OperatorWrapper
from ilastik.api.slotApi import (
    WrappedArrayLikeInputSlot, WrappedArrayLikeOutputSlot,
    WrappedValueSlot
)
from lazyflow.stype import ArrayLike, ValueLike


class TestwrappedLevelZeroSlotValueLike(object):
    def setup(self):
        self.op_pipe = OpValuePiper(graph=Graph())
        # Sanity checks:
        assert self.op_pipe.Input.level == 0
        assert self.op_pipe.Output.level == 0
        assert type(self.op_pipe.Input.stype) == ValueLike
        assert type(self.op_pipe.Input.stype) == ValueLike

    def test_emtpy_wrapping(self):
        op_pipe = self.op_pipe
        wrapped_input_slot = WrappedValueSlot(op_pipe.Input)
        assert wrapped_input_slot.slot == op_pipe.Input
        assert wrapped_input_slot._version == 0

    def test_value_setting(self):
        op_pipe = self.op_pipe
        wrapped_input_slot = WrappedValueSlot(op_pipe.Input)
        values = [
            'a value',
            1,
        ]
        for index, value in enumerate(values):
            wrapped_input_slot.set_value(value)
            assert op_pipe.Input.value == value, f"{op_pipe.Input.value[0]}: {value}"
            assert wrapped_input_slot._version == index + 1, (
                f"encountered {wrapped_input_slot._version}, expected {index + 1}")

    def test_value_getting(self):
        op_pipe = self.op_pipe
        wrapped_output_slot = WrappedValueSlot(op_pipe.Output)
        values = [
            'a value',
            1,
        ]
        for index, value in enumerate(values):
            op_pipe.Input.setValue(value)
            assert wrapped_output_slot._version == index + 1, (
                f"encountered {wrapped_input_slot._version}, expected {index + 1}")
            assert wrapped_output_slot.get_value() == value


class TestwrappedMultiLevelSlotValueLike(object):
    def setup(self):
        self.op_pipe = OperatorWrapper(
            OpValuePiper,
            graph=Graph()
        )
        # Sanity checks:
        assert self.op_pipe.Input.level == 1
        assert self.op_pipe.Output.level == 1
        assert type(self.op_pipe.Input.stype) == ValueLike
        assert type(self.op_pipe.Output.stype) == ValueLike

    def test_emtpy_wrapping(self):
        op_pipe = self.op_pipe
        wrapped_input_slot = WrappedValueSlot(op_pipe.Input)
        assert wrapped_input_slot.slot == op_pipe.Input
        assert wrapped_input_slot._version == 0, (
                f"encountered {wrapped_input_slot._version}, expected 0")

        wrapped_output_slot = WrappedValueSlot(op_pipe.Output)
        assert wrapped_output_slot.slot == op_pipe.Output
        assert wrapped_output_slot._version == 0, (
                f"encountered {wrapped_output_slot._version}, expected 0")

    def test_value_setting(self):
        op_pipe = self.op_pipe
        wrapped_input_slot = WrappedValueSlot(op_pipe.Input)
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
        wrapped_input_slot = WrappedValueSlot(op_pipe.Input)
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
        self.op_pipe.Output.stype = ArrayLike(self.op_pipe.Output)

        # Sanity checks
        assert type(self.op_pipe.Input.stype) == ArrayLike
        assert type(self.op_pipe.Output.stype) == ArrayLike
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
        assert wrapped_input_slot._version == 1, (
            f"Expected 1, but got {wrapped_input_slot._version}")
        # get output from wrapped slot
        wrapped_output = wrapped_output_slot.get(slicing=slicing, subindex=0)
        numpy.testing.assert_array_equal(wrapped_output, data_slice)

    def test_slot_multilevel_wrapping_w_content_axisorder(self):
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

        slicing_dict = {
            't': slice(0, 1),
            'c': slice(0, 2),
            'z': slice(0, 10),
            'y': slice(0, 20),
            'x': slice(0, 30)
        }

        original_slicing = tuple((slicing_dict[x] for x in default_axisorder))

        axisorders = ("".join(x) for x in itertools.permutations(default_axisorder))
        data_slice = data[original_slicing]

        tagged_data_slice = data_slice.view(vigra.VigraArray)
        tagged_data_slice.axistags = vigra.defaultAxistags(default_axisorder)

        expected_output = numpy.zeros((5, 2, 10, 40, 70), dtype=numpy.uint8)
        expected_output[original_slicing] = data_slice

        for index, axisorder in enumerate(axisorders):
            slicing = tuple((slicing_dict[x] for x in axisorder))
            transposed_data = tagged_data_slice.withAxes(*axisorder)
            wrapped_input_slot.write_into(
                slicing=slicing,
                data=transposed_data.view(numpy.ndarray),
                subindex=0,
                incoming_axis_order=axisorder
            )

            assert wrapped_input_slot._version == index + 1, (
                f"Expected {index + 1}, but got {wrapped_input_slot._version}")
            assert wrapped_input_slot._version == wrapped_output_slot._version, (
                f"Expected {wrapped_output_slot._version}, but got {wrapped_input_slot._version}")

            output = wrapped_output_slot.get(original_slicing, subindex=0)
            numpy.testing.assert_array_equal(output, data_slice)