###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2024, the ilastik developers
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
import collections
from functools import lru_cache, partial, wraps
import logging
import threading
from typing import Dict, Literal, Optional, Tuple, Union

import numpy
import vigra

import h5py

from lazyflow.graph import InputSlot, Operator, OutputSlot
from lazyflow.operators.generic import OpPixelOperator
from lazyflow.operators.opBlockedArrayCache import OpBlockedArrayCache
from lazyflow.operators.opCache import ManagedBlockedCache, MemInfoNode
from lazyflow.operators.opCacheFixer import OpCacheFixer
from lazyflow.operators.opReorderAxes import OpReorderAxes
from lazyflow.operators.opSimpleBlockedArrayCache import OpSimpleBlockedArrayCache
from lazyflow.operators.valueProviders import OpValueCache
from lazyflow.request.request import Request, RequestLock, RequestPool
from lazyflow.roi import roiToSlice
from lazyflow.slot import Slot
from lazyflow.utility import timeLogged, RamMeasurementContext


logger = logging.getLogger(__name__)


class OpRelabelConsecutive(Operator):
    Input = InputSlot()
    StartLabel = InputSlot(value=1)
    BypassModeEnabled = InputSlot(value=False)
    Output = OutputSlot()
    RelabelDict = OutputSlot()  # will always have only a "t" axis
    CachedRelabelDict = OutputSlot()  # will always have only a "t" axis
    CachedOutput = OutputSlot()
    CleanBlocks = OutputSlot()

    supportedDtypes = [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]

    def __init__(self, parent=None, graph=None):
        super().__init__(parent, graph)

        # internally we want a default order to simplify things
        self._op5 = OpReorderAxes(parent=self, Input=self.Input, AxisOrder="tzyxc")

        self._opDtypeConvert = OpPixelOperator(parent=self, Input=self._op5.Output, Function=lambda x: x)

        self._opRelabel = OpRelabelConsecutive5DNoCache(
            parent=self, Input=self._opDtypeConvert.Output, StartLabel=self.StartLabel
        )

        self._cache = OpBlockedArrayCache(parent=self)
        self._cache.name = "OpLabelVolume.OutputCache"
        self._cache.BypassModeEnabled.connect(self.BypassModeEnabled)
        self._cache.Input.connect(self._opRelabel.Output)
        self.CleanBlocks.connect(self._cache.CleanBlocks)

        self._relabel_dict_cache = OpBlockedArrayCache(parent=self)
        self._relabel_dict_cache.Input.connect(self._opRelabel.RelabelDict)
        self._relabel_dict_cache.Blockshape.setValue((1,))
        self.CachedRelabelDict.connect(self._relabel_dict_cache.Output)

        self._reoder_to_input_order = OpReorderAxes(parent=self, Input=self._opRelabel.Output, AxisOrder=None)
        self._reoder_to_input_order_cached = OpReorderAxes(parent=self, Input=self._cache.Output, AxisOrder=None)

        self.Output.connect(self._reoder_to_input_order.Output)
        self.RelabelDict.connect(self._opRelabel.RelabelDict)
        self.CachedOutput.connect(self._reoder_to_input_order_cached.Output)

    def setupOutputs(self):
        # check if the input dtype is valid
        if self.Input.ready():
            dtype = self.Input.meta.dtype
            if dtype not in self.supportedDtypes:
                msg = f"{self.name}: dtype '{dtype}' not supported. Supported types: {self.supportedDtypes}"
                raise ValueError(msg)

        # set cache chunk shape to the whole spatial volume
        shape = numpy.asarray(self._op5.Output.meta.shape, dtype=numpy.int64)
        shape[0] = 1
        shape[4] = 1
        self._cache.BlockShape.setValue(tuple(shape))

        # vigra cannot handle uint16 images in relabel - we'll convert
        # those automatically to uint32
        if self.Input.meta.dtype == numpy.uint16:
            self._opDtypeConvert.Function.setValue(lambda x: x.astype("uint32"))
        else:
            self._opDtypeConvert.Function.setValue(lambda x: x)

        # ensure Output is returned in the same order as input data
        input_order = self.Input.meta.getAxisKeys()
        self._reoder_to_input_order.AxisOrder.setValue(input_order)
        self._reoder_to_input_order_cached.AxisOrder.setValue(input_order)

    def propagateDirty(self, slot, subindex, roi):
        pass


class RelabelCache:

    name: str = "RelabelCache"

    def __init__(self, tagged_image_shape, startlabel=1):
        super().__init__()
        self.children = []
        self._tagged_shape = tagged_image_shape
        self._startlabel = startlabel
        self._block_locks = {i: threading.RLock() for i in range(tagged_image_shape["t"])}

        self._h5_handle = h5py.File(mode="a", name="test2", driver="core", backing_store=False)

        self._relabel_dict_cache = {}

        shape = tuple(val for val in tagged_image_shape.values())
        assert "t" in tagged_image_shape

        tagged_chunk_shape = {k: v for k, v in tagged_image_shape.items()}
        tagged_chunk_shape = {"t": 1, "c": 1, "z": 64, "y": 64, "x": 64}

        chunk_sizes = {ax: min(tagged_image_shape[ax], tagged_chunk_shape[ax]) for ax in tagged_image_shape.keys()}
        chunks = tuple(chunk_sizes.values())
        self._relabeled_image_cache = self._h5_handle.create_dataset(
            "relabel", shape=shape, chunks=chunks, compression="gzip"
        )

    def __call__(self, tagged_request_roi: Dict[Literal["t", "z", "y", "x", "c"], slice], label_image_slot: InputSlot):
        t = tagged_request_roi["t"]
        assert t.stop - t.start == 1
        with self._block_locks[t.start]:
            if t.start in self._relabel_dict_cache:
                return self._relabeled_image_cache[t.start], self._relabel_dict_cache[t.start]

            # do the computation first by expanding the roi
            whole_slice_request_roi = {k: v for k, v in tagged_request_roi.items()}
            for ax in "zyx":
                whole_slice_request_roi[ax] = slice(0, self._tagged_shape[ax])

            img = label_image_slot[tuple(whole_slice_request_roi.values())].wait().squeeze()
            breakpoint()
            _, __, labelmap_dict = vigra.analysis.relabelConsecutive(
                img, start_label=self._startlabel, keep_zeros=True, out=img
            )
            self._relabel_dict_cache[t.start] = labelmap_dict
            self._relabeled_image_cache[t.start] = img.withAxes("".join(tagged_request_roi.keys()))
            return img, labelmap_dict


class OpRelabelConsecutive5DNoCache(Operator):
    Input = InputSlot()  # in "tzyxc" order
    StartLabel = InputSlot()
    Output = OutputSlot()

    RelabelDict = OutputSlot()

    supportedDtypes = [numpy.uint8, numpy.uint32, numpy.uint64]

    def __init__(self, graph=None, parent=None, Input=None, StartLabel: Union[int, Slot] = 1):
        super().__init__(graph=graph, parent=parent)
        self.Input.setOrConnectIfAvailable(Input)
        self.StartLabel.setOrConnectIfAvailable(StartLabel)

        self._relable_dict = {}

    def setupOutputs(self):
        assert "".join(self.Input.meta.getAxisKeys()) == "tzyxc"
        self.Output.meta.assignFrom(self.Input.meta)
        self.RelabelDict.meta.shape = (self.Input.meta.getTaggedShape()["t"],)
        self.RelabelDict.meta.dtype = object
        self.RelabelDict.meta.axistags = vigra.defaultAxistags("t")

        self._cached_fun = RelabelCache(self.Input.meta.getTaggedShape(), startlabel=self.StartLabel.value)

    @timeLogged(logger)
    def execute(self, slot, subindex, roi, result):
        t_idx = self.Input.meta.getAxisKeys().index("t")

        tagged_shape = self.Input.meta.getTaggedShape()

        def relabel_single_slice(t, res_t):

            t_slice_roi = roi.copy()
            t_slice_roi.start[t_idx] = t
            t_slice_roi.stop[t_idx] = t + 1

            t_slc = roiToSlice(t_slice_roi.start, t_slice_roi.stop)

            label_image, label_dict = self._cached_fun(dict(zip(tagged_shape.keys(), t_slc)), self.Input)

            result_roi = roi.copy()
            result_roi.start[t_idx] = res_t
            result_roi.stop[t_idx] = res_t + 1
            if slot == self.RelabelDict:
                breakpoint()
                result[result_roi.toSlice()] = label_dict
            elif slot == self.Output:
                result[result_roi.toSlice()] = label_image.withAxes(self.Output.meta.axistags)

        pool = RequestPool()
        for res_t_ind, t in enumerate(range(roi.start[t_idx], roi.stop[t_idx])):
            pool.add(Request(partial(relabel_single_slice, t, res_t_ind)))

        pool.wait()

    def propagateDirty(self, slot, subindex, roi):

        self.Output.setDirty(())
        self.RelabelDict.setDirty(())
