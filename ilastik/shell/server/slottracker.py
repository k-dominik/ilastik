import collections
from functools import partial
import typing

from lazyflow.graph import OperatorWrapper
from lazyflow.operators.opReorderAxes import OpReorderAxes
from lazyflow.slot import Slot

import logging


logger = logging.getLogger(__name__)

VoxelSourceState = collections.namedtuple(
    "VoxelSourceState", "name axes shape dtype version")


class SlotTracker(object):
    """Copied from voxel_server"""

    def __init__(
            self,
            image_name_multislot: str,
            multislots: typing.List[Slot],
            forced_axes=None):
        """
        Args:
            image_name_multislot (str): should be
              a level 1 slot and hold a name for each image group (each lane)
            multislots (typing.List[Slot]): Outputs of OpReorderAxis for each of the multislots
            forced_axes (None, optional): Description
        """
        self.image_name_multislot = image_name_multislot
        self._slot_versions = {}  # { dataset_name : { slot_name : [slot, version] } }

        # TODO: name could not be enough
        self.multislot_names = [s.name for s in multislots]
        if forced_axes is None:
            self.multislots = multislots
        else:
            self.multislots = []
            for multislot in multislots:
                # HACK: skip slots with level > 1
                if multislot.level > 1:
                    logger.info(f'skipping slot {multislot} of {multislot.getRealOperator()}')
                    continue
                op = OperatorWrapper(
                    OpReorderAxes,
                    parent=multislot.getRealOperator().parent,
                    broadcastingSlotNames=['AxisOrder'])
                op.AxisOrder.setValue(forced_axes)
                op.Input.connect(multislot)
                self.multislots.append(op.Output)

    def get_dataset_names(self) -> typing.List[str]:
        names = []
        for lane_index, name_slot in enumerate(self.image_name_multislot):
            if name_slot.ready():
                name = name_slot.value
            else:
                name = "dataset-{}".format(lane_index)
            names.append(name)
        return names

    def get_slot_versions(self, dataset_name: str):
        found = False
        lane_index = None
        for lane_index, name_slot in enumerate(self.image_name_multislot):
            if name_slot.value == dataset_name:
                found = True
                break
        if not found:
            raise RuntimeError("Dataset name not found: {}".format(dataset_name))

        if dataset_name not in self._slot_versions:
            for multislot in self.multislots:
                slot = multislot[lane_index]
                # TODO: Unsubscribe to these signals on shutdown...
                slot.notifyDirty(partial(self._increment_version, dataset_name))
                slot.notifyMetaChanged(partial(self._increment_version, dataset_name))

            self._slot_versions[dataset_name] = {
                self.multislot_names[self.multislots.index(multislot)]: [multislot[lane_index], 0]
                for multislot in self.multislots}

        return self._slot_versions[dataset_name]

    def _increment_version(self, dataset_name: str, slot: Slot, *args) -> None:
        for name in self._slot_versions[dataset_name].keys():
            if self._slot_versions[dataset_name][name][0] == slot:
                self._slot_versions[dataset_name][name][1] += 1
                return
        assert False, "Couldn't find slot"

    def get_states(self, dataset_name: str):
        states = collections.OrderedDict()
        slot_versions = self.get_slot_versions(dataset_name)
        for slot_name, (slot, version) in slot_versions.items():
            try:
                axes = ''.join(slot.meta.getAxisKeys())
            except AssertionError:
                continue
            states[slot_name] = VoxelSourceState(slot_name,
                                                 axes,
                                                 slot.meta.shape,
                                                 slot.meta.dtype.__name__,
                                                 version)
        return states

    def get_slot(self, dataset_name: str, slot_name: str) -> Slot:
        slot_versions = self.get_slot_versions(dataset_name)
        return slot_versions[slot_name][0]