from lazyflow.graph import Graph, InputSlot, OutputSlot, Operator
from lazyflow.stype import ArrayLike, ImageType, ValueLike
from lazyflow.operators import OpArrayPiper
from ilastik.applets.base.standardApplet import StandardApplet
from ilastik.api.appletApi import WrappedTLO
from ilastik.api.slotApi import (
    WrappedSlot, WrappedArrayLikeInputSlot, WrappedArrayLikeOutputSlot
)


class OpMockTLO(Operator):
    # Inputs
    ValueInput = InputSlot(stype=ValueLike)
    ValueInputBroad = InputSlot(stype=ValueLike)
    ImageTypeInput = InputSlot(stype=ImageType)
    ArrayLikeInput = InputSlot(stype=ArrayLike)
    ImageTypeInputConnected = InputSlot(stype=ImageType, optional=True)
    # Slot with default value
    ValueInputDefault = InputSlot(stype=ValueLike, value=('default_set',))

    # Outputs
    ValueOutput = OutputSlot(stype=ValueLike)
    ImageTypeOutput = OutputSlot(stype=ImageType)
    ArrayLikeOutput = OutputSlot(stype=ArrayLike)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.opAP1 = OpArrayPiper(parent=self)
        self.opAP2 = OpArrayPiper(parent=self)

        self.opAP1.Input.connect(self.ImageTypeInput)
        self.ImageTypeOutput.connect(self.opAP1.Output)

        self.opAP2.Input.connect(self.ArrayLikeInput)
        self.ArrayLikeOutput.connect(self.opAP2.Output)

    def setupOutputs(self):
        self.ValueOutput.meta.assignFrom(self.ValueInput.meta)
        self.ValueOutput.setValue(self.ValueInput.value)

    def propagateDirty(self, slot, subindex, roi):
        # Check for proper name because subclasses may define extra inputs.
        # (but decline to override notifyDirty)
        self.ImageTypeOutput.setDirty(slice(None))
        self.ArrayLikeOutput.setDirty(slice(None))

    def setInSlot(self, slot, subindex, roi, value):
        # Implementations of this method is only needed to satisfy the flow of
        # the __setitem__ method for input slots. Nothing needs to be done here
        # as the input of the value slot is manipulated directly. When the
        # output is requested, execute is called.
        assert subindex == ()
        assert (slot == self.ImageTypeInput) or (slot == self.ArrayLikeInput)


class MockApplet(StandardApplet):
    def __init__(self, name, workflow):
        self._topLevelOperator = OpMockTLO(graph=Graph())
        super().__init__(name, workflow)

    @property
    def singleLaneOperatorClass(self):
        """
        Return the operator class which handles a single image.
        Single-lane applets should override this property.
        (Multi-lane applets must override ``topLevelOperator`` directly.)
        """
        return OpMockTLO

    @property
    def broadcastingSlots(self):
        """
        Slots that should be connected to all image lanes are referred to as "broadcasting" slots.
        Single-lane applets should override this property to return a list of the broadcasting slots' names.
        (Multi-lane applets must override ``topLevelOperator`` directly.)
        """
        return ['ValueInputBroad']


class MockWorkflow(Operator):
    def __init__(self):
        super().__init__(graph=Graph())


class TestAppletWrapping(object):
    def setup(self):
        self.workflow = MockWorkflow()
        self.applet = MockApplet(name='MockApplet', workflow=self.workflow)

    def test_appletWrapping(self):
        incoming_axis_order = 'tczyx'
        outgoing_axis_order = 'tczyx'
        wrapped_tlo = WrappedTLO(
            self.applet,
            input_axis_order=incoming_axis_order,
            output_axis_order=outgoing_axis_order
        )

        assert wrapped_tlo.name == self.applet.name

        input_slots = wrapped_tlo._input_slots
        output_slots = wrapped_tlo._output_slots

        assert input_slots is not None
        assert output_slots is not None

        print(input_slots)

        assert all(isinstance(x['slot'], WrappedSlot) for x in input_slots.values())
        assert all(isinstance(x['slot'], WrappedSlot) for x in output_slots.values())

        assert all(
            x['slot'].axis_order == incoming_axis_order
            for x in input_slots.values() if isinstance(x['slot'], WrappedArrayLikeInputSlot))
        assert all(
            x['slot'].axis_order == incoming_axis_order
            for x in output_slots.values() if isinstance(x['slot'], WrappedArrayLikeOutputSlot))

        assert not any(x.lower().find("arraylike") != -1 for x in input_slots), (
            "Should not wrap plain ArrayLike slots.")

        assert not any(x.lower().find("arraylike") != -1 for x in output_slots), (
            "Should not wrap plain ArrayLike slots.")

