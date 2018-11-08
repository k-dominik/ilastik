from lazyflow.graph import Graph, InputSlot, OutputSlot, Operator
from lazyflow.stype import ArrayLike, ImageType, ValueSlotType
from ilastik.applets.base.standardApplet import StandardApplet
from ilastik.api.appletApi import WrappedTLO


class OpMockTLO(Operator):
    # Inputs
    ValueInput = InputSlot(stype=ValueSlotType)
    ValueInputBroad = InputSlot(stype=ValueSlotType)
    ImageTypeInput = InputSlot(stype=ImageType)
    ArrayLikeInput = InputSlot(stype=ArrayLike)

    # Slot with default value
    ValueInputDefault = InputSlot(stype=ValueSlotType, value=('default_set',))

    # Outputs
    ValueOutput = OutputSlot(stype=ValueSlotType)
    ImageTypeOutput = OutputSlot(stype=ImageType)
    ArrayLikeOutput = OutputSlot(stype=ArrayLike)

    def setupOutputs(self):
        self.ImageTypeOutput.meta.assignFrom(self.ImageTypeInput.meta)
        self.ArrayLikeOutput.meta.assignFrom(self.ArrayLikeInput.meta)
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
        wrapped_tlo = WrappedTLO(self.applet)

        assert wrapped_tlo.name == self.applet.name
