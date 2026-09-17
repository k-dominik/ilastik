###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2026, the ilastik developers
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
# 		   http://ilastik.org/license.html
###############################################################################
import logging
import warnings
from itertools import islice
from math import ceil
from threading import RLock
from typing import Callable, Iterable, Literal, Optional, TypeVar

import numpy as np
import numpy.typing as npt
import torch
import vigra
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize

from ilastik.applets.objectFeatureCollection.types import EmbeddingVector
from ilastik.config import runtime_cfg
from ilastik.plugins import ObjectFeaturesPlugin
from ilastik.plugins.types import FeatureDescription, FloatArray, PartialFeatureDict, PluginInfo
from lazyflow.base import ItemId
from lazyflow.operators.ioOperators.opFileListToGridReader import FileList
from lazyflow.utility.grid import _OUTPUT_AXIS_KEYS

T = TypeVar("T")


logger = logging.getLogger(__name__)


def get_preferred_device():
    device_id = runtime_cfg.preferred_cuda_device_id

    if not device_id:
        device_id = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

    logger.info(f"Neural Network Workflow: using default device {device_id}")

    return torch.device(device_id)


# From python docs - added to python in 3.12
def batched(iterable: Iterable[T], n: int, *, strict: bool = False) -> Iterable[Iterable[T]]:
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


DynoModelFamily = Literal["dinov2"]

DINO_MODELS: dict[DynoModelFamily, dict[str, str]] = {
    "dinov2": {
        "small": "dinov2_vits14",
        "base": "dinov2_vitb14",
        "large": "dinov2_vitl14",
        "giant": "dinov2_vitg14",
        "small-reg": "dinov2_vits14_reg",
        "base-reg": "dinov2_vitb14_reg",
        "large-reg": "dinov2_vitl14_reg",
        "giant-reg": "dinov2_vitg14_reg",
    },
}


def processor():
    return Compose(
        [
            Resize(
                256,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            CenterCrop(224),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


class DinoMeta(type):
    def __new__(metacls, name, bases, namespace, dino_model_family: DynoModelFamily, dino_model_variant: str):
        return super().__new__(metacls, name, bases, namespace)

    def __init__(cls, name, bases, dct, dino_model_family: DynoModelFamily, dino_model_variant: str):
        cls.dino_model_family = dino_model_family
        cls.dino_model_variant = dino_model_variant

        cls.plugin_info = PluginInfo(
            name=f"Dyno Object Features {dino_model_family}:{dino_model_variant}",
            author="ilastik team",
            version="0.1",
            description="todo",
        )


class DinoBase(ObjectFeaturesPlugin):
    _processor: Compose = processor()
    _model = None
    dino_model_family: str
    dino_model_variant: str
    lock = RLock()

    def __init__(self):
        self.model_id = DINO_MODELS[self.dino_model_family][self.dino_model_variant]
        self._safe_model_id = self.model_id.replace("/", ".'")
        self._processor: Compose = processor()

        self._batch_size = 32

        self.device = get_preferred_device()
        print(f"{self.device=}, {runtime_cfg.preferred_cuda_device_id=}")

        self._feature_dict = {
            f"{self._safe_model_id}": FeatureDescription(
                displaytext=f"{self.dino_model_family}:{self.dino_model_variant}",
                detailtext="...",
                advanced=False,
                tooltip="todo",
                group="intensity",
                no_3D=True,
                channel_aware=True,
            )
        }

    @classmethod
    def _ensure_model(cls, model_id: str):
        if not cls._model:
            cls._model = torch.hub.load(
                f"facebookresearch/{cls.dino_model_family}:main",
                model_id,
                pretrained=True,
            )

    def availableFeatures(self, image: vigra.VigraArray, labels: vigra.VigraArray) -> dict[str, dict[str, str]]:
        if labels.ndim != 2:
            return {}

        return self._feature_dict

    def compute_global(
        self, image: vigra.VigraArray, labels: vigra.VigraArray, features: PartialFeatureDict, axes: str
    ) -> dict[str, FloatArray]:

        # the image parameter passed here is the whole dataset.
        # We can use it estimate if the data is 2D or 3D and then apply
        # this knowledge in compute_local
        nZ = image.shape[axes.z]
        if nZ > 1:
            self.ndim = 3
        else:
            self.ndim = 2

        return self._do_4d(image, labels, features, axes)

    def _do_4d(self, image: vigra.VigraArray, labels: vigra.VigraArray, features: PartialFeatureDict, axes: str):
        result = vigra.analysis.extractRegionFeatures(
            image.squeeze().astype(np.float32),
            labels.squeeze().astype(np.uint32),
            ["Coord<Maximum>", "Coord<Minimum>"],
            ignoreLabel=0,
        )

        # Todo: extract region around
        # 1: ignore background object
        coords_min = result["Coord<Minimum>"][1:]
        coords_max = result["Coord<Maximum>"][1:]

        slicings = [
            self.compute_extent(image, c_min, c_max, axes, margin=(0, 0, 0))
            for c_min, c_max in zip(coords_min, coords_max)
        ]

        n_objects = coords_min.shape[0]

        feats = []
        for start in range(0, n_objects, self._batch_size):
            print(f"batch {start=}")
            feats.extend(self._compute_batch(image, slicings[start : start + self._batch_size], start, axes))

        return {f"{self._safe_model_id}": np.array(feats)}

    @staticmethod
    def _ensure_rgb(
        image_array: npt.NDArray[np.uint8], axistags: tuple[str, ...] = _OUTPUT_AXIS_KEYS
    ) -> npt.NDArray[np.float32]:
        tagged_shape = dict(zip(axistags, image_array.shape))
        if any(s for s, v in tagged_shape.items() if s not in "xyc" and v > 1):
            raise NotImplementedError("Dino Features can only be computed in 2D")
        if "c" not in tagged_shape:
            axistags = axistags + ("c",)
            tagged_shape["c"] = 1
            image_array = image_array[:, np.newaxis]
        assert tagged_shape["c"] <= 3
        c_index = axistags.index("c")
        if image_array.shape[c_index] == 1:
            arr_rgb = np.concatenate([image_array, image_array, image_array], axis=c_index)
        elif image_array.shape[c_index] == 2:
            slicing = [slice(None) for _ in axistags]
            slicing[c_index] = slice(0, 1)
            arr_rgb = np.concatenate([image_array, image_array[tuple(slicing)]], axis=c_index)

        # has 3 channels
        # ensure channel axis first
        trans = (
            axistags.index("c"),
            axistags.index("y"),
            axistags.index("x"),
            axistags.index("z"),
        )
        return torch.Tensor(arr_rgb.transpose(trans).squeeze()) / 255.0

    def compute_batchwise(
        self,
        image_dict: FileList,
        progress_callback: Optional[Callable[[float], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[ItemId, EmbeddingVector]:
        def nop(_arg: float | str):
            pass

        if first_offending := next((x for x in image_dict.values() if np.dtype(x.dtype) != np.uint8), None):
            warnings.warn(f"Data is expected to be uint8! Got {first_offending=}")
        message_callback = message_callback or nop
        progress_callback = progress_callback or nop

        progress_callback(-1)

        all_features: dict[ItemId, EmbeddingVector] = {}
        n_items = len(image_dict)
        if n_items == 0:
            logger.info("No images to process")
            return all_features

        n_batches = ceil(n_items / self._batch_size)

        with self.lock:
            type(self)._ensure_model(self.model_id)
            assert self._model
            self._model.to(self.device)
            progress_callback(0.0)
            for batch_index, batch in enumerate(batched(image_dict.items(), self._batch_size)):
                batch_ids = tuple(b[0] for b in batch)
                images = [self._ensure_rgb(image_info.accessor()) for id, image_info in batch]
                inputs = [self._processor(image) for image in images]

                inputs = torch.stack(inputs).to(
                    self.device,
                    non_blocking=True,
                )
                with torch.inference_mode():
                    features = self._model(inputs)

                all_features.update(
                    {
                        image_id: EmbeddingVector(id=image_id, embedding_vector=vec)
                        for image_id, vec in zip(
                            batch_ids, features.detach().cpu().numpy().astype(np.float32, copy=False)
                        )
                    }
                )
                message_callback(f"Embedding batch {batch_index} of {n_batches}")
                progress_callback((batch_index + 1) / n_batches * 100)

            message_callback("Embedding done.")
            progress_callback(100.0)

        return all_features

    def _compute_batch(self, image, extents, start, axes):
        with self.lock:
            print("start")
            type(self)._ensure_model(self.model_id)

            assert self._model
            # TODO: FIXME: hardcoded axes - will not work ;)
            images = [
                self._ensure_rgb(self.compute_rawbbox(image, extent, axes), axistags="txyzc") for extent in extents
            ]
            inputs = [self._processor(img) for img in images]
            self._model.to(self.device)
            # synchronize(device)

            with torch.inference_mode():
                features = self._model(inputs)

            # synchronize(device)
            # assert features
            return features.detach().cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def compute_extent(image, mincoords, maxcoords, axes, margin):
        """Make a slicing to extract object i from the image."""
        # find the bounding box (margin is always 'xyz' order)
        result = [None] * 3
        minx = int(max(mincoords[axes.x] - margin[axes.x], 0))
        miny = int(max(mincoords[axes.y] - margin[axes.y], 0))

        # Coord<Minimum> and Coord<Maximum> give us the [min,max]
        # coords of the object, but we want the bounding box: [min,max), so add 1
        maxx = int(min(maxcoords[axes.x] + 1 + margin[axes.x], image.shape[axes.x]))
        maxy = int(min(maxcoords[axes.y] + 1 + margin[axes.y], image.shape[axes.y]))

        result[axes.x] = slice(minx, maxx)
        result[axes.y] = slice(miny, maxy)

        try:
            minz = max(mincoords[axes.z] - margin[axes.z], 0)
            maxz = min(maxcoords[axes.z] + 1 + margin[axes.z], image.shape[axes.z])
        except:
            minz = 0
            maxz = 1

        result[axes.z] = slice(int(minz), int(maxz))

        return result

    def compute_rawbbox(self, image, extent, axes):
        """essentially returns image[extent], preserving all channels."""
        key = [x for x in extent]
        key.insert(axes.c, slice(None))
        # HACK: make rgb
        return np.concatenate([image[tuple(key)], image[tuple(key)], image[tuple(key)]])

    def fill_properties(self, feature_dict):
        """Augment die feature dictionary with additional fields

        For every feature in the feature dictionary, fill in its properties,
        such as 'detailtext', which will be displayed in help, or 'displaytext'
        which will be displayed instead of the feature name

        Only necessary if these keys are not present to begin with.

        Args:
            feature_dict: list of feature names

        Returns:
            same dictionary, with additional fields filled for each feature

        """
        for k, v in feature_dict.items():
            v.update(self._feature_dict[k])
        return feature_dict


class DinoV2Base(DinoBase):
    dino_model_family = "dinov2"
    dino_model_variant = "base"

    plugin_info = PluginInfo(
        name=f"dinov2:base",
        author="ilastik team",
        version="0.1",
        description="todo",
    )
    pass


class DinoV2BaseReg(DinoBase):
    dino_model_family = "dinov2"
    dino_model_variant = "base-reg"

    plugin_info = PluginInfo(
        name=f"dinov2:base-reg",
        author="ilastik team",
        version="0.1",
        description="todo",
    )
    pass


class DinoV2Small(DinoBase):
    dino_model_family = "dinov2"
    dino_model_variant = "small"

    plugin_info = PluginInfo(
        name=f"dinov2:small",
        author="ilastik team",
        version="0.1",
        description="todo",
    )
    pass


class DinoV2SmallReg(DinoBase):
    dino_model_family = "dinov2"
    dino_model_variant = "small-reg"

    plugin_info = PluginInfo(
        name=f"dinov2:small-reg",
        author="ilastik team",
        version="0.1",
        description="todo",
    )
    pass


class DinoV2Large(DinoBase):
    dino_model_family = "dinov2"
    dino_model_variant = "large"

    plugin_info = PluginInfo(
        name=f"dinov2:large",
        author="ilastik team",
        version="0.1",
        description="todo",
    )
    pass


class DinoV2LargeReg(DinoBase):
    dino_model_family = "dinov2"
    dino_model_variant = "large-reg"

    plugin_info = PluginInfo(
        name=f"dinov2:large-reg",
        author="ilastik team",
        version="0.1",
        description="todo",
    )
    pass
