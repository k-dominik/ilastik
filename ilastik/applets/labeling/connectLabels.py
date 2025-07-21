###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2025, the ilastik developers
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
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from itertools import product
from typing import List, Literal, Optional, Tuple, Union

import vigra
from vigra.analysis import extractRegionFeatures

from lazyflow.request.request import Request, RequestPool
from lazyflow.utility.io_util.write_ome_zarr import SPATIAL_AXES


class Neighbourhood(IntEnum):
    NONE = 0
    SINGLE = 1  # Axis aligned, 2D: 4, 3D: 6
    NDIM = 2  # Full block, 2D: 8, 3D: 26


class BlockBoundary(IntEnum):
    NONE = 0
    START = 1
    STOP = 2


BoundaryDescrRelative = dict[Literal["x", "y", "z"], BlockBoundary]
BoundaryDescr = dict[Literal["x", "y", "z"], Union[int, None]]


@dataclass(slots=True, frozen=True)
class Region:
    axistags: str
    slices: Tuple[slice, ...]
    label: int

    @property
    def tagged_slicing(self):
        # zip here hides that axistags and slices might be out of sync, in fact axistags have a c....
        return dict(zip(self.axistags, self.slices))

    @property
    def tagged_center(self):
        return {k: int((sl.stop - sl.start) / 2) + sl.start for k, sl in self.tagged_slicing.items()}

    def is_at_boundary(self, boundary: BoundaryDescr) -> bool:
        if all(b is None for b in boundary.values()):
            return False

        is_at_boundary = True
        for k, coord in boundary.items():
            if coord is not None:
                sl = self.tagged_slicing[k]
                if sl.start == coord or sl.stop == coord:
                    continue
                else:
                    return False

        return is_at_boundary

    @property
    def key(self):
        return Tuple((sl.start, sl.stop) for sl in self.slices)

    @classmethod
    def with_slices(cls, reg: "Region", slices: Tuple[slice, ...]) -> "Region":
        assert len(slices) == len(reg.slices)
        return Region(axistags=reg.axistags, slices=slices, label=reg.label)


def add_tagged_coords(t1, t2):
    assert set(t1.keys()).issubset(set(t2.keys())), f"{list(t1.keys())=}, {list(t2.keys())=}"
    return {k: t1[k] + t2[k] for k in t1.keys()}


@dataclass(frozen=True)
class Block:
    axistags: str
    slices: Tuple[slice, ...]
    regions: List[Region]
    neigbourhood: Neighbourhood = Neighbourhood.NDIM

    @property
    def key(self) -> Tuple[int, ...]:
        return Tuple(sl.start for sl in self.slices)

    @property
    def tagged_slices(self):
        return {tag: sl for tag, sl in zip(self.axistags, self.slices)}

    @property
    def spatial_axes(self):
        return [x for x in self.axistags if x in SPATIAL_AXES]

    @property
    def tagged_start(self):
        return {tag: sl.start for tag, sl in zip(self.axistags, self.slices)}

    def boundary_regions(self, boundary: BoundaryDescrRelative, label: Optional[int] = None):
        if self.neigbourhood == Neighbourhood.NONE:
            return

        def boundary_index_from_slice(sl: slice, boundary: BlockBoundary) -> Union[int, None]:
            if boundary == BlockBoundary.NONE:
                return None
            if boundary == BlockBoundary.START:
                return 0
            if boundary == BlockBoundary.STOP:
                return sl.stop - sl.start

        tagged_boundary = {}
        for at, bd in boundary.items():
            tagged_boundary[at] = boundary_index_from_slice(self.tagged_slices[at], bd)

        def labelmatch(region_label) -> bool:
            if label == None:
                return True
            else:
                return region_label == label

        for region in self.regions:
            if region.is_at_boundary(tagged_boundary) and labelmatch(region.label):
                yield region

    def boundaries_positive(self):
        n_spatial = len(self.spatial_axes)
        boundary_iter = product([BlockBoundary.NONE, BlockBoundary.STOP], repeat=n_spatial)

        for boundary in boundary_iter:
            if all(x == BlockBoundary.NONE for x in boundary):
                continue

            yield dict(zip(self.spatial_axes, boundary))

    def boundary_regions_positive(self):
        for boundary in self.boundaries_positive():
            for boundary_region in self.boundary_regions(boundary):
                yield boundary, boundary_region

    def neighbour_start_coordinates(self, boundary: BoundaryDescrRelative):
        tagged_slices = self.tagged_slices

        neighbour_start = {}
        for k, sl in tagged_slices.items():
            if k not in SPATIAL_AXES:
                neighbour_start[k] = sl.start
                continue

            assert k in boundary
            b = boundary[k]
            if b == BlockBoundary.NONE:
                neighbour_start[k] = sl.start
            elif b == BlockBoundary.START:
                neighbour_start[k] = sl.start - (sl.stop - sl.start)
            elif b == BlockBoundary.STOP:
                neighbour_start[k] = sl.stop
            else:
                # unreachable
                raise NotImplemented()

        return tuple(neighbour_start[k] for k in self.axistags)

    def region_to_world(self, region: Region):
        tagged_region_sl = region.tagged_slicing
        world_sl = {}
        for k, sl in tagged_region_sl.items():
            if k in SPATIAL_AXES:
                world_sl[k] = slice(sl.start + self.tagged_start[k], sl.stop + self.tagged_start[k])
            else:
                world_sl[k] = sl
        # hack, currently axistags seem to be out of sync and contain c
        return region.with_slices(region, tuple(world_sl[k] for k in region.axistags if k in world_sl))


def extract_annotations(axistags: str, labels_data) -> Tuple[Region, ...]:
    if "z" in axistags and labels_data.depth > 1:
        connected_components = vigra.analysis.labelVolumeWithBackground(
            labels_data.astype("uint32"),
            neighborhood=26,
        )
    else:
        connected_components = vigra.analysis.labelImageWithBackground(
            labels_data.astype("uint32"),
            neighborhood=8,
        )
    feats = extractRegionFeatures(
        labels_data.astype("float32"),
        connected_components,
        ignoreLabel=0,
        features=["RegionCenter", "Coord<Maximum>", "Coord<Minimum>", "Minimum"],
    )

    # shape: (n_objs, ndim)
    max_bb = feats["Coord<Maximum>"].astype("uint32") + 1
    min_bb = feats["Coord<Minimum>"].astype("uint32")

    slices: list[Tuple[slice, ...]] = []
    for min_, max_ in zip(min_bb, max_bb):
        slices.append(tuple(slice(mi, ma) for mi, ma in zip(min_, max_)))
    # we pass the label image as "image", so minimum will be the same label
    labels = feats["Minimum"].astype("uint32")

    regions = Tuple(Region(axistags=axistags, slices=sl, label=label) for sl, label in zip(slices[1::], labels[1::]))

    return regions


def connect_regions(non_zero_slicings, label_slot, axistags):
    regions: dict[Tuple[Tuple[int, int], ...], Region] = {}
    blocks: dict[Tuple[int, ...], Block] = {}

    regions_dict: dict[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]] = {}

    def extract_single(roi):
        labels_data = vigra.taggedView(label_slot[roi].wait(), "".join(axistags))
        block_regions = extract_annotations(axistags, labels_data)
        block = Block(axistags="".join(axistags), slices=roi, regions=block_regions)
        blocks[block.key] = block

    pool = RequestPool()
    for roi in non_zero_slicings:
        pool.add(Request(partial(extract_single, roi)))

    pool.wait()
    pool.clean()

    def invert_boundary(boundary: dict[str, BlockBoundary]) -> dict[str, BlockBoundary]:
        # TODO: do this with a dict
        b = {}
        for k, d in boundary.items():
            if d == BlockBoundary.NONE:
                b[k] = BlockBoundary.NONE
            elif d == BlockBoundary.START:
                b[k] = BlockBoundary.STOP
            elif d == BlockBoundary.STOP:
                b[k] = BlockBoundary.START

        return b

    def get_anchor(region):
        if region.key not in regions_dict:
            return region.key

        current = region.key
        while True:
            anchor = regions_dict[current]
            if anchor == current:
                return anchor
            current = anchor

    for block in blocks.values():
        for region in block.regions:
            region_world = block.region_to_world(region)
            regions[region_world.key] = region_world
            if region_world.key not in regions_dict:
                regions_dict[region_world.key] = region_world.key
        for boundary, region in block.boundary_regions_positive():
            region_world = block.region_to_world(region)
            block_start = block.neighbour_start_coordinates(boundary)
            if block_start not in blocks:
                continue

            neighbour_block = blocks[block_start]
            boundary_in_neighbour = invert_boundary(boundary)
            for reg in neighbour_block.boundary_regions(boundary_in_neighbour, label=region.label):
                neighbour_region_world = neighbour_block.region_to_world(reg)
                if check_overlap(region_world, neighbour_region_world):
                    anchor_neighbour = get_anchor(neighbour_region_world)
                    anchor_reg = get_anchor(region_world)

                    regions_dict[anchor_neighbour] = anchor_reg

    return regions_dict, regions


def check_overlap(region_a: Region, region_b: Region) -> bool:
    assert region_a.axistags == region_b.axistags

    overlap = True
    for k, v in region_a.tagged_slicing.items():
        if k not in SPATIAL_AXES:
            continue
        if not (v.stop >= region_b.tagged_slicing[k].start and region_b.tagged_slicing[k].stop >= v.start):
            return False

    return overlap
