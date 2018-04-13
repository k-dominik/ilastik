import numpy
import vigra

from lazyflow.operators.opArrayPiper import OpArrayPiper
from lazyflow.graph import Graph, InputSlot, OutputSlot
from lazyflow.operatorWrapper import OperatorWrapper
from ilastik.shell.server.slotApi import WrappedInputSlot, WrappedOutputSlot
from lazyflow.stype import ArrayLike

class OpArrayPiperArrayLike(OpArrayPiper):
    name = "ArrayPiperArrayLike"
    description = "simple piping operator with 'array-like' input/output"

    #Inputs
    Input = InputSlot(allow_mask=True, stype=ArrayLike)
   
    #Outputs
    Output = OutputSlot(allow_mask=True, stype=ArrayLike)


class TestWrappedLevelZeroSlotArrayLike(object):
    def setup(self):
        self.op_pipe = OpArrayPiper(graph=Graph())

        # Sanity checks:
        assert self.op_pipe.Input.level == 0
        assert self.op_pipe.Output.level == 0

    def test_slot_wrapping_empty(self):
        op_pipe = self.op_pipe

        wrapped_input_slot = WrappedInputSlot(op_pipe.Input, incoming_axis_order='tczyx')
        wrapped_output_slot = WrappedOutputSlot(op_pipe.Output, forced_axisorder='tczyx')


class TestWrappedMultilevelSlotArrayLike(object):
    def setup(self):
        self.op_pipe = OperatorWrapper(
            OpArrayPiperArrayLike,
            graph=Graph()
        )

        # Sanity checks
        assert self.op_pipe.Input.level == 1
        assert self.op_pipe.Output.level == 1
        assert len(self.op_pipe.Input) == 0
        assert len(self.op_pipe.Input) == 0


    def test_slot_wrapping_empty(self):
        op_pipe = self.op_pipe

        wrapped_input_slot = WrappedInputSlot(op_pipe.Input, incoming_axis_order='tczyx')
        wrapped_output_slot = WrappedOutputSlot(op_pipe.Output, forced_axisorder='tczyx')

        assert wrapped_output_slot.axis_order == 'tczyx'
        assert wrapped_input_slot.axis_order == 'tczyx'

    def test_slot_multilevel_wrapping_w_content(self):
        op_pipe = self.op_pipe
        data = numpy.random.randint(
            0, 255, (5, 2, 10, 40, 70), dtype=numpy.uint8)

        op_pipe.Input.resize(2)
        assert len(op_pipe.Input) == 2
        assert len(op_pipe.Input) == 2


        wrapped_input_slot = WrappedInputSlot(op_pipe.Input)
        wrapped_output_slot = WrappedOutputSlot(op_pipe.Output)

        assert wrapped_input_slot.slot.level == 1
        assert wrapped_output_slot.slot.level == 1

        wrapped_input_slot[0].meta.axistags = vigra.defaultAxistags('tczyx')
        wrapped_input_slot[0].setValue(numpy.zeros((5, 2, 10, 40, 70), dtype=numpy.uint8))

        slicing = (
            slice(0, 1),    # t
            slice(0, 3),    # c
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

        output = wrapped_input_slot[0][:].wait()
        expected_output = numpy.zeros((5, 2, 10, 40, 70), dtype=numpy.uint8)
        expected_output[slicing] = data_slice
        numpy.testing.assert_array_equal(output, expected_output)

    def test_slot_multilevel_wrapping_w_content(self):
        op_pipe = self.op_pipe
        data = numpy.random.randint(
            0, 255, (5, 2, 10, 40, 70), dtype=numpy.uint8)

        op_pipe.Input.resize(2)
        assert len(op_pipe.Input) == 2
        assert len(op_pipe.Input) == 2


        wrapped_input_slot = WrappedInputSlot(op_pipe.Input)
        wrapped_output_slot = WrappedOutputSlot(op_pipe.Output)

        assert wrapped_input_slot.slot.level == 1
        assert wrapped_output_slot.slot.level == 1

        wrapped_input_slot[0].meta.axistags = vigra.defaultAxistags('tczyx')
        wrapped_input_slot[0].setValue(numpy.zeros((5, 2, 10, 40, 70), dtype=numpy.uint8))

        slicing = (
            slice(0, 1),    # t
            slice(0, 3),    # c
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

        output = wrapped_input_slot[0][:].wait()
        expected_output = numpy.zeros((5, 2, 10, 40, 70), dtype=numpy.uint8)
        expected_output[slicing] = data_slice
        expected_output 
        numpy.testing.assert_array_equal(output, expected_output)
