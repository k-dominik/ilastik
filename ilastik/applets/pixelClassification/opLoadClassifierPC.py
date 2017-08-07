###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# In addition, as a special exception, the copyright holders of
# ilastik give you permission to combine ilastik with applets,
# workflows and plugins which are not covered under the GNU
# General Public License.
#
# See the LICENSE file for details. License information is also available
# on the ilastik web site at:
#          http://ilastik.org/license.html
###############################################################################
import logging

import numpy

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.operators import OpReorderAxes, OpBlockedArrayCache

loggger = logging.getLogger('__file__')


class OpSimpleAddNoiseOperator(Operator):
    InputImage = InputSlot()

    MinValue = InputSlot(value=0, optional=True)
    MaxValue = InputSlot(value=10, optional=True)

    OutputImage = OutputSlot()

    def setupOutputs(self):
        self.OutputImage.meta.shape = self.InputImage.meta.shape
        self.OutputImage.meta.shape = self.InputImage.meta.dtype
        self.OutputImage.meta.axistags = self.InputImage.meta.axistags

    def propagateDirty(self, slot, subindex, roi):
        self.output.setDirty(roi)

    def execute(self, slot, subindex, roi, result):
        a = self.InputImage.get(roi).wait()
        noise = numpy.random.randint(
            self.MinValue, self.MaxValue, a.shape, self.InputImage.meta.dtype)
        result[...] = a + noise

        return result


class OpLoadClassifierPixelClassification(Operator):
    """

    Internal AxisOrder for this Operator: 'tczyx'. Is enforced by reorder axes.

    """
    # Input Slots
    RawInput = InputSlot(optional=True)  # For display
    InputImage = InputSlot()

    # Output Slots
    Output = OutputSlot()  # Continuous access
    CachedOutput = OutputSlot()  # block-wise access

    def __init__(self, *args, **kwargs):
        super(OpLoadClassifierPixelClassification, self).__init__(*args, **kwargs)

        # Reorient image to make subsequent slicing easier
        self.opReorderInput = OpReorderAxes(parent=self)
        self.opReorderInput.AxisOrder.setValue('tzyxc')
        self.opReorderInput.Input.connect(self.InputImage)

        self.opFilter = OpSimpleAddNoiseOperator(parent=self)
        self.opFilter.InputImage.connect(self.opReorderInput.Output)

        self.opReorderOutput = OpReorderAxes(parent=self)
        self.opReorderOutput.Input.connect(self.opFilter.OutputImage)

        self.Output.connect(self.opReorderOutput.Output)

        self.opCache = OpBlockedArrayCache(parent=self)
        self.opCache.CompressionEnabled.setValue(True)
        self.opCache.Input.connect(self.opReorderOutput.Output)

        self.CachedOutput.connect(self.opCache.Output)
