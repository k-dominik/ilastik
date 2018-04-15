from lazyflow.graph import Graph, InputSlot, OutputSlot, Operator

from lazyflow.stype import ArrayLike, ImageType, ValueSlotType

from ilastik.applets.base.standardApplet import StandardApplet

from ilastik.shell.server.appletApi import WrappedApplet

from ilastik.workflow import Workflow

class OpDummyTLO(Operator):
    # Inputs
    ValueInput = InputSlot(stype=ValueSlotType)
    ValueInputBroad = InputSlot(stype=ValueSlotType)
    ImageTypeInput = InputSlot(stype=ImageType)
    ArrayLikeInput = InputSlot(stype=ArrayLike)

    # Slot with default value
    ValueInputDefault = InputSlot(stype=ValueSlotType, value='default_set')

    # Outputs
    ValueOutput = OutputSlot(stype=ValueSlotType)
    ImageTypeOutput = OutputSlot(stype=ImageType)
    ArrayLikeOutput = OutputSlot(stype=ArrayLike)

    def setupOutputs(self):
        self.ImageTypeOutput.meta.assignFrom(ImageTypeInput.meta)
        self.ArrayLikeOutput.meta.assignFrom(ArrayLikeInput.meta)
        self.ValueOutput.setValue(self.ValueInput.value)

    def propagateDirty(self, slot, subindex, roi):
        key = roi.toSlice()
        # Check for proper name because subclasses may define extra inputs.
        # (but decline to override notifyDirty)
        if slot.name == 'ImageTypeInput':
            self.ImageTypeOutput.setDirty(slice(None))
        elif slot.name == 'ArrayLikeInput':
            self.ArrayLikeOutput.setDirty(slice(None))


    def setInSlot(self, slot, subindex, roi, value):
        # Implementations of this method is only needed to satisfy the flow of
        # the __setitem__ method for input slots. Nothing needs to be done here
        # as the input of the value slot is manipulated directly. When the
        # output is requested, execute is called.
        assert subindex == ()
        assert (slot == self.ImageTypeInput) or (slot == self.ArrayLikeInput)

class DummyApplet(StandardApplet):
    def __init__(self):
        self._topLevelOperator = OpDummyTLO(graph=Graph())
        super().__init__('DummyApplet')

    @property
    def singleLaneOperatorClass(self):
        """
        Return the operator class which handles a single image.
        Single-lane applets should override this property.
        (Multi-lane applets must override ``topLevelOperator`` directly.)
        """
        return OpDummyTLO

    @property
    def broadcastingSlots(self):
        """
        Slots that should be connected to all image lanes are referred to as "broadcasting" slots.
        Single-lane applets should override this property to return a list of the broadcasting slots' names.
        (Multi-lane applets must override ``topLevelOperator`` directly.)
        """
        return ['ValueInputBroad']


class DummyWorkfow()


class TestAppletWrapping(object):
    def setup(self):
        self.applet = DummyApplet()

    def test_appletWrapping(self):
        
        wrapped_applet = WrappedApplet(self.applet)        

        print(wrapped_applet)
        assert False