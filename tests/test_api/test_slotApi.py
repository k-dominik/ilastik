import numpy

from lazyflow.operators.opArrayPiper import OpArrayPiper
from lazyflow.graph import Graph
from lazyflow.operatorWrapper import OperatorWrapper
from ilastik.shell.server.slotApi import WrappedInputSlot, WrappedOutputSlot


class TestWrappedApplet(object):

    def test_slot_wrapping_empty(self):
        op_pipe = OperatorWrapper(
            OpArrayPiper,
            graph=Graph()
        )
        print(op_pipe.parent)

        assert op_pipe.Output.level == 1
        assert len(op_pipe.Input) == 0
        assert len(op_pipe.Input) == 0

        wrapped_input_slot = WrappedInputSlot(op_pipe.Input, incoming_axis_order='tczxy')
        wrapped_output_slot = WrappedOutputSlot(op_pipe.Output, forced_axisorder='tczxy')

        assert wrapped_output_slot.axis_order == 'tczxy'
        assert wrapped_input_slot.axis_order == 'tczxy'


    def test_slot_wrapping_w_content(self):
        data = numpy.random.randint(
            0, 255, (5, 2, 10, 40, 70), dtype=numpy.uint8)
        op_pipe = OperatorWrapper(
            OpArrayPiper,
            graph=Graph()
        )

        assert op_pipe.Output.level == 1
        assert len(op_pipe.Input) == 0
        assert len(op_pipe.Input) == 0

        op_pipe.Input.resize(2)
        assert len(op_pipe.Input) == 2
        assert len(op_pipe.Input) == 2


        wrapped_input_slot = WrappedInputSlot(op_pipe.Input)
        wrapped_output_slot = WrappedOutputSlot(op_pipe.Output)

        assert wrapped_input_slot.slot.level == 1
        assert wrapped_output_slot.slot.level == 1

        slicing = (
            slice(0, 1),    # t
            slice(None),    # c
            slice(0, 10),   # z
            slice(0, 20),   # y
            slice(0, 30)    # x
        )

        data_slice = data[slicing]

        wrapped_input_slot.write_into(
            slicing,
            data_slice
        )
