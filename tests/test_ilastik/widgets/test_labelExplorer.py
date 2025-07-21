import numpy
import pytest
import vigra

from ilastik.applets.labeling.connectLabels import (
    Block,
    BlockBoundary,
    BoundaryDescrRelative,
    Region,
    extract_annotations,
)


@pytest.mark.parametrize(
    "boundary",
    [
        {"x": BlockBoundary.START, "y": BlockBoundary.NONE},
        {"x": BlockBoundary.STOP, "y": BlockBoundary.NONE},
        {"x": BlockBoundary.NONE, "y": BlockBoundary.START},
        {"x": BlockBoundary.NONE, "y": BlockBoundary.STOP},
        {"x": BlockBoundary.START, "y": BlockBoundary.START},
        {"x": BlockBoundary.STOP, "y": BlockBoundary.START},
        {"x": BlockBoundary.START, "y": BlockBoundary.STOP},
        {"x": BlockBoundary.STOP, "y": BlockBoundary.STOP},
    ],
    ids=["left", "right", "top", "bottom", "top-left", "top-right", "bottom-left", "bottom_right"],
)
def test_no_region_at_boundary_2d(boundary: BoundaryDescrRelative):
    axistags = "yx"
    r1 = Region(axistags=axistags, slices=(slice(1, 8), slice(1, 8)), label=1)
    r2 = Region(axistags=axistags, slices=(slice(1, 2), slice(1, 2)), label=1)
    r3 = Region(axistags=axistags, slices=(slice(5, 9), slice(1, 2)), label=2)

    b = Block(axistags=axistags, slices=(slice(10, 20), slice(20, 30)), regions=(r1, r2, r3))

    assert len(list(b.boundary_regions(boundary))) == 0


@pytest.mark.parametrize(
    "boundary, expected_regions",
    [
        ({"x": BlockBoundary.START, "y": BlockBoundary.NONE}, 3),
        ({"x": BlockBoundary.STOP, "y": BlockBoundary.NONE}, 3),
        ({"x": BlockBoundary.NONE, "y": BlockBoundary.START}, 3),
        ({"x": BlockBoundary.NONE, "y": BlockBoundary.STOP}, 3),
        ({"x": BlockBoundary.START, "y": BlockBoundary.START}, 1),
        ({"x": BlockBoundary.STOP, "y": BlockBoundary.START}, 1),
        ({"x": BlockBoundary.START, "y": BlockBoundary.STOP}, 1),
        ({"x": BlockBoundary.STOP, "y": BlockBoundary.STOP}, 1),
    ],
    ids=["left", "right", "top", "bottom", "top-left", "top-right", "bottom-left", "bottom_right"],
)
def test_boundary_regions_2d(boundary: BoundaryDescrRelative, expected_regions):
    # TODO: test that the correct regions are returned!
    axistags = "xy"

    r_left = Region(axistags=axistags, slices=(slice(0, 9), slice(1, 2)), label=1)
    r_right = Region(axistags=axistags, slices=(slice(1, 10), slice(1, 2)), label=1)
    r_top = Region(axistags=axistags, slices=(slice(3, 6), slice(0, 4)), label=2)
    r_bottom = Region(axistags=axistags, slices=(slice(2, 4), slice(1, 10)), label=42)

    r_topleft = Region(axistags=axistags, slices=(slice(0, 3), slice(0, 2)), label=3)
    r_topright = Region(axistags=axistags, slices=(slice(5, 10), slice(0, 4)), label=4)

    r_bottomleft = Region(axistags=axistags, slices=(slice(0, 3), slice(4, 10)), label=6)
    r_bottomright = Region(axistags=axistags, slices=(slice(1, 10), slice(5, 10)), label=7)

    b = Block(
        axistags=axistags,
        slices=(slice(10, 20), slice(20, 30)),
        regions=(r_left, r_right, r_top, r_bottom, r_topleft, r_topright, r_bottomleft, r_bottomright),
    )

    assert len(list(b.boundary_regions(boundary))) == expected_regions


@pytest.mark.parametrize(
    "boundary, label, expected_regions",
    [
        ({"x": BlockBoundary.START, "y": BlockBoundary.NONE}, 1, 1),
        ({"x": BlockBoundary.STOP, "y": BlockBoundary.NONE}, 1, 1),
        ({"x": BlockBoundary.NONE, "y": BlockBoundary.START}, 2, 2),
        ({"x": BlockBoundary.NONE, "y": BlockBoundary.STOP}, 42, 1),
        ({"x": BlockBoundary.START, "y": BlockBoundary.START}, 2, 1),
        ({"x": BlockBoundary.STOP, "y": BlockBoundary.START}, 4, 1),
        ({"x": BlockBoundary.START, "y": BlockBoundary.STOP}, 6, 1),
        ({"x": BlockBoundary.STOP, "y": BlockBoundary.STOP}, 7, 1),
    ],
    ids=["left", "right", "top", "bottom", "top-left", "top-right", "bottom-left", "bottom_right"],
)
def test_boundary_regions_per_label_2d(boundary, label, expected_regions):
    # TODO: test that the correct regions are returned!
    axistags = "xy"

    r_left = Region(axistags=axistags, slices=(slice(0, 9), slice(1, 2)), label=1)
    r_right = Region(axistags=axistags, slices=(slice(1, 10), slice(1, 2)), label=1)
    r_top = Region(axistags=axistags, slices=(slice(3, 6), slice(0, 4)), label=2)
    r_bottom = Region(axistags=axistags, slices=(slice(2, 4), slice(1, 10)), label=42)

    r_topleft = Region(axistags=axistags, slices=(slice(0, 3), slice(0, 2)), label=2)
    r_topright = Region(axistags=axistags, slices=(slice(5, 10), slice(0, 4)), label=4)

    r_bottomleft = Region(axistags=axistags, slices=(slice(0, 3), slice(4, 10)), label=6)
    r_bottomright = Region(axistags=axistags, slices=(slice(1, 10), slice(5, 10)), label=7)

    b = Block(
        axistags=axistags,
        slices=(slice(10, 20), slice(20, 30)),
        regions=(r_left, r_right, r_top, r_bottom, r_topleft, r_topright, r_bottomleft, r_bottomright),
    )

    assert len(list(b.boundary_regions(boundary, label=label))) == expected_regions


def test_extract_annotations():
    axistags = "xy"

    data = numpy.zeros((100, 100), dtype="uint32")
    data[0:1, :] = 1
    data[3:5, 3:7] = 2
    data[8:9, 8:10] = 1

    labels_data = vigra.taggedView(data, axistags=axistags)

    regions = extract_annotations(axistags=axistags, labels_data=labels_data)

    assert len(regions) == 3
    assert len([r for r in regions if r.label == 1]) == 2
    assert len([r for r in regions if r.label == 2]) == 1

    r_2 = next((r for r in regions if r.label == 2))
    assert r_2.axistags == axistags
    assert r_2.tagged_slicing["x"] == slice(3, 5)
    assert r_2.tagged_slicing["y"] == slice(3, 7)
