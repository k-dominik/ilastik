"""
Todos:

- [ ] embedding projection update on embedding fine tuning
- [ ] DataSelection: show entries from file list - not separate lanes
- [ ] Need to have a pool for file-pointers
- [ ] Add table export
- [ ] Add batch processing
- [ ] serialization of embedding network / offline access
- [ ] typing: 1. rename FileList to FileTable. Then also have a "bare" FileTable
      class without counter and the likes - this is only needed in dataselection.
- [ ] pipe through the embedding backbone from feature computation
- [ ] check if there are enough images for the selected efforts
- [ ]

Done:
- [x] ui: live update button and state tracking
- [x] add serialization
  - [x] embedding
  - [x] classifier
  - [x] umap
  - [x] labels
  - [x] label names
  - [x] label colors
  - [x] pmap colors
- [x] embedding projection checkbox:
  - [x] show labels
  - [x] show predictions
- [x] Move feature computation out to separate applet
- [x] consider making OpGridLabels two separate ops / one for label display, one for handling the labeling
- [x] Secondary Controls Widget
  - [x] Embedding on top
  - [x] bottom another volumina, that only shows labelled objects
        op: Inputs FileList and Labels, embeds opgrid, and exposes grid image and labelimage output
        probably all there
- [x] save axistags in FileList
- [x] embedding projection: Add multi-selection
- [x] sync colors in label view
- [x] add fine tuning for embedding
  - [x] add controls
    - [x] choose effort
    - [x] start / stop button
    - [x] progress
- [x] come up with something sensible for adaptation effort...
- [x] Check labels deserialization - seems odd - I think there's more
      labels after deserialization and the names are gone
- [x] fix progress updates during adaptation

Defer:
- [ ] download status / progress for dino models
- [ ] ui to select which base model to use
- [ ] at some point: future: deal with deletions of datasets from file list (e.g. labels should be dropped)
- [ ] Add compute features (morphometrics)

Notes:
* In general operators are not expected to change their metadata (such as shape) without a disconnect/connect
  (dirtiness is not part oft that), so in order to work with that, the grid is disconnected/connected when a change
  (e.g. subselection) is detected so that the intended flow of setupOutputs -> shape change can take place in a
  coordinated way (all slots that source Grid must be connected/disconnected)
"""

import logging
import random
import warnings
from enum import IntEnum
from functools import partial
from itertools import islice
from math import ceil
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, TypeVar, cast

import numpy as np
import numpy.typing as npt
import torch
import umap
import vigra
from pydantic.dataclasses import dataclass
from qtpy.QtGui import QColor
from sklearn.preprocessing import StandardScaler

from ilastik.applets.fileCollection.fileCollectionOps import OpGrid, OpGridView
from ilastik.applets.objectClassification.opObjectClassification import InvalidObjectIndex
from ilastik.applets.objectClassificationCollection.adaptembedding import DinoV2Backbone, Projector, adapt_features
from ilastik.applets.objectFeatureCollection.types import EmbeddingTable, EmbeddingVector
from ilastik.config import runtime_cfg
from lazyflow import USER_LOGLEVEL
from lazyflow.base import ItemId
from lazyflow.classifiers.parallelVigraRfLazyflowClassifier import (
    ParallelVigraRfLazyflowClassifier,
    ParallelVigraRfLazyflowClassifierFactory,
)
from lazyflow.operator import Operator
from lazyflow.operators.generic import OpSelectSubslot
from lazyflow.operators.ioOperators.types import FileList as FileListT
from lazyflow.operators.ioOperators.types import FileListDataRow, RowBase
from lazyflow.operators.opSlicedBlockedArrayCache import OpSlicedBlockedArrayCache
from lazyflow.operators.valueProviders import OpValueCache
from lazyflow.rtype import Index
from lazyflow.slot import InputSlot, OutputSlot, Slot
from lazyflow.stype import Opaque
from lazyflow.utility.grid import _OUTPUT_AXIS_KEYS, ImageGrid
from lazyflow.utility.orderedSignal import OrderedSignal

from .types import LabelRow, ProjectorData, UmapRow, UmapTable

if TYPE_CHECKING:
    from lazyflow.graph import Graph
    from lazyflow.rtype import Roi
    from lazyflow.utility.grid import ImageGrid

    from .types import AdaptionParameters as AdaptionParametersT
    from .types import LabelTable as LabelTableT


logger = logging.getLogger(__name__)


user_log = partial(logger.log, USER_LOGLEVEL)


def get_preferred_device():
    device_id = runtime_cfg.preferred_cuda_device_id

    if not device_id:
        device_id = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

    logger.info(f"Neural Network Workflow: using default device {device_id}")

    return torch.device(device_id)


class EmbeddingSource(IntEnum):
    Original = 0
    Adapted = 1


class OpGridLabels(Operator):
    Input = InputSlot(optional=True, rtype=Index, stype=Opaque)
    LabelTable = OutputSlot["LabelTableT"](stype="object")

    def __init__(self, graph: Optional["Graph"] = None, parent: Optional[Operator] = None):
        super().__init__(graph=graph, parent=parent)
        self._lock = Lock()
        # this is only to adhere to the cache interface for serialization
        self._dirty = False
        # label table has object_id as index
        self._label_table: "LabelTableT" = {}

    def setupOutputs(self):
        self._label_table = {}

        self.LabelTable.meta.shape = (1,)
        self.LabelTable.meta.dtype = object

    def propagateDirty(self, slot, subindex, roi):
        assert slot in [self.Input]
        self.LabelTable.setDirty()

    def execute(self, slot, subindex, roi, result):
        assert slot in [self.LabelTable]
        return [self._label_table]

    def _setInSlot(self, slot: InputSlot, subindex: int, roi: "Index", value: Any):
        assert slot == self.Input
        image_id = ItemId(roi._pslice)
        assert self._label_table is not None
        self._label_table[image_id] = LabelRow(id=image_id, label=value)
        self.LabelTable.setDirty(())
        # probably should connect to FileList -> get notified if files added/removed
        # which can then be cross checked with label table and updated accordingly

    def forceValue(self, value: "LabelTableT", set_dirty: bool = True):
        """Meant for deserialization"""
        with self._lock:
            self._label_table = value
            self._dirty = False
        if set_dirty:
            self.LabelTable.setDirty()


class OpGridLabelImage(Operator):
    Grid = InputSlot["ImageGrid"]()
    LabelTable = InputSlot["LabelTableT"]()

    GridLabels = OutputSlot()

    def __init__(
        self,
        parent: Optional[Operator] = None,
        graph: Optional["Graph"] = None,
        Grid: Optional[Slot["ImageGrid"]] = None,
        LabelTable: Optional[Slot["LabelTableT"]] = None,
    ):
        super().__init__(graph=graph, parent=parent)
        self.Grid.setOrConnectIfAvailable(Grid)
        self.LabelTable.setOrConnectIfAvailable(LabelTable)

    def setupOutputs(self):
        self._grid: ImageGrid = self.Grid.value

        output_shape = dict(zip(self._grid.output_axis_keys, self._grid.output_shape))
        output_shape["c"] = 1
        self.GridLabels.meta.shape = tuple(output_shape[k] for k in self._grid.output_axis_keys)
        self.GridLabels.meta.dtype = np.dtype("uint8").type
        self.GridLabels.meta.axistags = vigra.defaultAxistags("".join(self._grid.output_axis_keys))

    def execute(self, slot: OutputSlot, subindex, roi: "Roi", result: npt.NDArray[np.uint8]):
        assert slot == self.GridLabels
        block = np.zeros_like(result)
        label_table = self.LabelTable.value
        for grid_cell in self._grid.grid_cells_per_roi(roi):
            val = 0
            if grid_cell.image_index is None:
                continue
            try:
                if grid_cell.image_index in label_table:
                    val = label_table[grid_cell.image_index].label
            except KeyError:
                pass
            data = grid_cell.data(dtype=result.dtype, fill_value=val)
            data_inner = grid_cell.data(dtype=result.dtype, fill_value=255, additional_margin=4)
            data[data_inner == 255] = 0
            block[grid_cell.roi_local_clamped_slice] = data
        result[:] = block

    def propagateDirty(self, slot: InputSlot, subindex, roi):
        assert slot in [self.Grid, self.LabelTable]
        if slot == self.Grid:
            self.setupOutputs()
        self.GridLabels.setDirty()


class OpEmbeddingAdapt(Operator):
    FileList = InputSlot["FileListT"](stype="object")
    Input = InputSlot["LabelTableT"](stype="object")
    ModelId = InputSlot[str](value="dinov2_vits14_reg")
    AdaptionParameters = InputSlot["AdaptionParametersT"](stype="object")

    Projector = OutputSlot[ProjectorData](stype="object")

    def __init__(
        self,
        parent: Optional[Operator] = None,
        graph: Optional["Graph"] = None,
        FileList: Optional[Slot["FileListT"]] = None,
        Input: Optional[Slot["LabelTableT"]] = None,
        AdaptionParameters: Optional[Slot["AdaptionParametersT"]] = None,
    ):
        super().__init__(parent=parent, graph=graph)
        self.FileList.setOrConnectIfAvailable(FileList)
        self.Input.setOrConnectIfAvailable(Input)
        self.AdaptionParameters.setOrConnectIfAvailable(AdaptionParameters)
        self.progress_signal = OrderedSignal()
        self.cancellation_token: CancellationToken | None = None

    def setupOutputs(self):
        self.Projector.meta.shape = (1,)
        self.Projector.meta.dtype = object

    def execute(self, slot, subindex, roi, result):
        assert slot == self.Projector
        adaption_parameters = self.AdaptionParameters.value
        print(f"Requesting projector with {adaption_parameters=}")

        PROJ_HIDDEN_DIM: int = 256
        PROJ_OUT_DIM: int = 128
        device = get_preferred_device()
        backbone = DinoV2Backbone(model_name=self.ModelId.value).to(device)
        backbone.eval()
        with torch.no_grad():
            backbone_dim = backbone(torch.zeros(1, 3, 224, 224, device=device)).shape[1]
        projector = Projector(backbone_dim, PROJ_HIDDEN_DIM, PROJ_OUT_DIM).to(device)

        # Adapt the features
        print("Adapting features ...")
        items = self.FileList.value
        labels = self.Input.value
        subselected_items = self._prepare_items(items, labels, adaption_parameters.n_unlabeled)
        num_classes = max(label.label for label in labels.values())
        history = adapt_features(
            subselected_items,
            labels,
            num_classes,
            backbone,
            projector,
            device,
            labeled_fraction=adaption_parameters.labeled_fraction,
            cancellation_token=self.cancellation_token,
            progress_callback=self.progress_signal,
            message_callback=user_log,
        )

        # ensure we're "done" in any case
        self.progress_signal(100)

        projector_data = ProjectorData(
            state_dict=projector.state_dict(),
            adaptation_parameters=adaption_parameters,
            input_dim=projector.input_dim,
            hidden_dim=projector.hidden_dim,
            output_dim=projector.output_dim,
            n_epochs=history[-1]["epoch"],
        )

        return [projector_data]

    def _prepare_items(self, items: "FileListT", labels: "LabelTableT", n_unlabeled: int) -> "FileListT":
        unlabeled = [k for k in items if k not in labels or labels[k].label == 0]
        if len(unlabeled) < n_unlabeled:
            raise ValueError(
                f"Number of unlabeled objects {len(unlabeled)=} is smaller than requested number {n_unlabeled=}"
            )

        subselected_items = {k: items[k] for k in random.sample(unlabeled, k=n_unlabeled)}
        really_labeled = [k.id for k in labels.values() if k.label != 0]
        subselected_items.update({k: items[k] for k in really_labeled})
        return subselected_items

    def propagateDirty(self, slot, subindex, roi):
        self.Projector.setDirty()


class OpObjectCollectionTrain(Operator):
    Labels = InputSlot["LabelTableT"]()
    NumClasses = InputSlot(stype="int")
    Features = InputSlot[EmbeddingTable](stype="object")
    ForestCount = InputSlot(stype="int", value=1)

    Classifier = OutputSlot[ParallelVigraRfLazyflowClassifier]()

    def __init__(
        self,
        parent: Optional[Operator] = None,
        graph: Optional["Graph"] = None,
        Labels: Optional[Slot["LabelTable"]] = None,
        NumClasses=None,
        Features=None,
    ):
        super().__init__(parent=parent, graph=graph)
        self._tree_count = 100
        self.Labels.setOrConnectIfAvailable(Labels)
        self.NumClasses.setOrConnectIfAvailable(NumClasses)
        self.Features.setOrConnectIfAvailable(Features)

        self._classifier = None

    def setupOutputs(self):
        self.Classifier.meta.dtype = object
        self.Classifier.meta.shape = (1,)
        self.Classifier.meta.axistags = None

    def execute(self, slot, subindex, roi, result):

        features = self.Features.value
        labels = self.Labels.value
        labels_nonzero = {k: v for k, v in labels.items() if v.label != 0}
        del labels

        if len(labels_nonzero) == 0:
            result[0] = None
            return result

        all_labels = list(range(1, self.NumClasses.value + 1))

        labels_arr = np.array([row.label for row in labels_nonzero.values()], dtype="uint32")
        features_arr = np.array([features[row.id].embedding_vector for row in labels_nonzero.values()], dtype="float32")

        classifier_factory = ParallelVigraRfLazyflowClassifierFactory(
            self._tree_count, self.ForestCount.value, labels=all_labels
        )
        classifier = classifier_factory.create_and_train(features_arr, labels_arr)
        avg_oob = np.mean(classifier.oobs)
        logger.info("training finished, average out-of-bag error: {}".format(avg_oob))
        result[0] = classifier
        return result

    def propagateDirty(self, slot, subindex, roi):
        self.Classifier.setDirty(())


@dataclass
class PredictionRow(RowBase):
    prediction: int
    probabilities: tuple[float, ...]
    uncertainty_estimate: float


PredictionTable = dict[ItemId, PredictionRow]


class OpObjectCollectionPredict(Operator):

    Features = InputSlot[EmbeddingTable]()
    Classifier = InputSlot[ParallelVigraRfLazyflowClassifier](stype="object")

    Predictions = OutputSlot[PredictionTable](stype="object")

    def __init__(
        self, parent: Optional[Operator] = None, graph: Optional["Graph"] = None, Features=None, Classifier=None
    ):
        super().__init__(parent, graph)
        self.Features.setOrConnectIfAvailable(Features)
        self.Classifier.setOrConnectIfAvailable(Classifier)

    def setupOutputs(self):
        self.Predictions.meta.shape = self.Features.meta.shape
        self.Predictions.meta.dtype = object
        self.Predictions.meta.axistags = None

    def execute(self, slot: OutputSlot[PredictionTable], subindex, roi, result):
        assert slot == self.Predictions
        classifier = self.Classifier.value
        if classifier is None:
            return [{}]

        features = self.Features.value
        feature_array = np.array([f.embedding_vector for f in features.values()], dtype="float32")

        prob_predictions: npt.NDArray[np.float32] = cast(
            npt.NDArray[np.float32], classifier.predict_probabilities(feature_array)
        )

        predictions = prob_predictions.argmax(axis=-1) + 1
        sorted_probs = np.sort(prob_predictions, axis=-1)
        uncertainties = 1 - sorted_probs[:, -1] + sorted_probs[:, -2]

        return [
            {
                id: PredictionRow(id=id, probabilities=tuple(probs), prediction=pred, uncertainty_estimate=uncertainty)
                for id, probs, pred, uncertainty in zip(features.keys(), prob_predictions, predictions, uncertainties)
            }
        ]

    def propagateDirty(self, slot, subindex, roi):
        self.Predictions.setDirty(())


class OpGridPredictionsImage(Operator):
    Grid = InputSlot["ImageGrid"]()
    Predictions = InputSlot[PredictionTable]()

    GridPredictions = OutputSlot()

    def __init__(
        self,
        graph: Optional["Graph"] = None,
        parent: Optional[Operator] = None,
        Grid: Optional[Slot["ImageGrid"]] = None,
        Predictions: Optional[Slot[PredictionTable]] = None,
    ):
        super().__init__(graph=graph, parent=parent)
        self.Grid.setOrConnectIfAvailable(Grid)
        self.Predictions.setOrConnectIfAvailable(Predictions)

    def setupOutputs(self):
        self._grid: ImageGrid = self.Grid.value
        output_shape = dict(zip(self._grid.output_axis_keys, self._grid.output_shape))
        output_shape["c"] = 1
        self.GridPredictions.meta.shape = tuple(output_shape[k] for k in self._grid.output_axis_keys)
        self.GridPredictions.meta.dtype = np.dtype("uint8").type
        self.GridPredictions.meta.axistags = vigra.defaultAxistags("".join(self._grid.output_axis_keys))

    def propagateDirty(self, slot, subindex, roi):
        assert slot in [self.Grid, self.Predictions]
        if slot == self.Grid:
            self.setupOutputs()
        self.GridPredictions.setDirty()

    def execute(self, slot: OutputSlot, subindex, roi: "Roi", result: npt.NDArray[np.uint8]):
        assert slot in [self.GridPredictions]
        predictions = self.Predictions.value
        block = np.zeros_like(result)
        for grid_cell in self._grid.grid_cells_per_roi(roi):
            val = 0
            if grid_cell.image_index is None:
                continue
            try:
                if grid_cell.image_index in predictions:
                    val = predictions[grid_cell.image_index].prediction
            except KeyError:
                pass
            data = grid_cell.data(dtype=result.dtype, fill_value=val)
            block[grid_cell.roi_local_clamped_slice] = data
        result[:] = block

    def propagateDirty(self, slot, subindex, roi):
        self.GridPredictions.setDirty()


class OpUmap(Operator):
    Features = InputSlot[EmbeddingTable](stype="object")
    Umap = OutputSlot[UmapTable](stype="object")

    def __init__(
        self,
        graph: Optional["Graph"] = None,
        parent: Optional[Operator] = None,
        Features: Optional[Slot[EmbeddingTable]] = None,
        write_logs: bool = False,
    ):
        super().__init__(parent=parent, graph=graph, write_logs=write_logs)
        self.Features.setOrConnectIfAvailable(Features)
        self.progress_signal = OrderedSignal()

    def setupOutputs(self):
        self.Umap.meta.dtype = "object"
        self.Umap.meta.shape = (1,)
        self.Umap.meta.axistags = None

    def propagateDirty(self, slot, subindex, roi):
        self.Umap.setDirty()

    def execute(self, slot, subindex, roi, result):
        assert slot == self.Umap
        self.progress_signal(-1)
        try:
            features = self.Features.value
            feature_vector = [f.embedding_vector for f in features.values()]
            df_scaled = StandardScaler().fit_transform(feature_vector)

            reducer = umap.UMAP()
            embedding = reducer.fit_transform(df_scaled)
            return [{id: UmapRow(id=id, x=e[0], y=e[1]) for id, e in zip(features.keys(), embedding)}]
        finally:
            self.progress_signal(100)


_T = TypeVar("_T")


# From python docs - added to python in 3.12
def batched(iterable: Iterable[_T], n: int, *, strict: bool = False) -> Iterable[Iterable[_T]]:
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


class OpEmbedding(Operator):
    Embedding = InputSlot["EmbeddingTable"](stype="object")
    FineTunedProjectorData = InputSlot[ProjectorData](stype="object")

    ProjectedEmbedding = OutputSlot[EmbeddingTable](stype="object")

    def __init__(
        self,
        parent: Optional[Operator] = None,
        graph: Optional["Graph"] = None,
        Embedding: Optional[Slot["EmbeddingTable"]] = None,
        FineTunedProjectorData: Optional[Slot["ProjectorData"]] = None,
    ):
        super().__init__(parent=parent, graph=graph)
        self.Embedding.setOrConnectIfAvailable(Embedding)
        self.FineTunedProjectorData.setOrConnectIfAvailable(FineTunedProjectorData)
        self._table: dict["ItemId", FileListDataRow] = {}
        self.progress_signal = OrderedSignal()

    def setupOutputs(self):
        self.ProjectedEmbedding.meta.shape = (1,)
        self.ProjectedEmbedding.meta.dtype = object

    def execute(self, slot, subindex, roi, result):
        assert slot == self.ProjectedEmbedding
        projector_data = self.FineTunedProjectorData.value
        projector = Projector(
            input_dim=projector_data.input_dim,
            hidden_dim=projector_data.hidden_dim,
            output_dim=projector_data.output_dim,
        )
        projector.load_state_dict(projector_data.state_dict)

        return [
            self._compute_batchwise(
                projector, self.Embedding.value, progress_callback=self.progress_signal, message_callback=user_log
            )
        ]

    def _compute_batchwise(
        self,
        projector: Projector,
        feature_table: "EmbeddingTable",
        progress_callback: Optional[Callable[[float], None]] = None,
        message_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[ItemId, EmbeddingVector]:
        def nop(_arg: float | str):
            pass

        message_callback = message_callback or nop
        progress_callback = progress_callback or nop
        progress_callback(-1)

        all_features: dict[ItemId, EmbeddingVector] = {}
        n_items = len(feature_table)
        if n_items == 0:
            logger.info("No images to process")
            return all_features

        BATCH_SIZE = 64
        n_batches = ceil(n_items / BATCH_SIZE)

        device = get_preferred_device()
        print(f"Running on {device=}")
        assert projector
        projector.to(device)
        progress_callback(0.0)
        for batch_index, batch in enumerate(batched(feature_table.items(), BATCH_SIZE)):
            batch_ids = tuple(b[0] for b in batch)
            raw_embeddings = [torch.Tensor(embedding_vector.embedding_vector) for id, embedding_vector in batch]

            inputs = torch.stack(raw_embeddings).to(
                device,
                non_blocking=True,
            )
            with torch.inference_mode():
                features = projector(inputs)

            all_features.update(
                {
                    image_id: EmbeddingVector(id=image_id, embedding_vector=vec)
                    for image_id, vec in zip(
                        batch_ids, features[0].detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                }
            )
            message_callback(f"Embedding batch {batch_index} of {n_batches}")
            progress_callback((batch_index + 1) / n_batches * 100)

        message_callback("Embedding done.")
        progress_callback(100.0)

        return all_features

    def propagateDirty(self, slot, subindex, roi):
        self.ProjectedEmbedding.setDirty()


class OpLabelView(Operator):
    FileList = InputSlot[FileListT]()
    AnnotationsTable = InputSlot["LabelTableT"]()

    GridImage = OutputSlot()
    GridLabels = OutputSlot()

    def __init__(
        self,
        parent: Optional[Operator] = None,
        graph: Optional["Graph"] = None,
        FileList: Optional[Slot[FileListT]] = None,
        AnnotationsTable: Optional[Slot["LabelTableT"]] = None,
    ):
        super().__init__(parent, graph)
        self.FileList.setOrConnectIfAvailable(FileList)
        self.AnnotationsTable.setOrConnectIfAvailable(AnnotationsTable)

        self.op_grid = OpGrid(parent=self, all_if_no_subselection=False)
        self.op_grid.name = "OpLabelView-OpGrid"

        self.op_grid_image = OpGridView(parent=self, FileList=self.FileList)
        self.op_grid_image.name = "OpLabelView-OpGrid"
        self.GridImage.connect(self.op_grid_image.Output)

        self.op_grid_label_view = OpGridLabelImage(parent=self)
        self.op_grid_label_view.LabelTable.connect(self.AnnotationsTable)
        self.GridLabels.connect(self.op_grid_label_view.GridLabels)
        self.GridImage.connect(self.op_grid_image.Output)

        def update_subselection(*args, **kwargs):
            labels_dict = self.AnnotationsTable.value
            indices = [x.id for x in sorted(labels_dict.values(), key=lambda x: x.label) if x.label != 0]
            self.op_grid.SubsetObjects.setValue(indices)

        self.AnnotationsTable.notifyDirty(update_subselection)

        def update_grid(*args):
            self.GridImage.disconnect()
            self.GridLabels.disconnect()
            if self.op_grid.Grid.ready():
                grid = self.op_grid.Grid.value
                print(f"LabelViewOp: {grid} {len(self.AnnotationsTable.value)=}")
                self.op_grid_image.Grid.setValue(grid)
                self.op_grid_label_view.Grid.setValue(grid)

                self.GridImage.connect(self.op_grid_image.Output)
                self.GridLabels.connect(self.op_grid_label_view.GridLabels)

        # for when subselection changes
        self.op_grid.Grid.notifyDirty(update_grid)
        # for the initial connect in connectLane
        self.op_grid.Grid.notifyReady(update_grid)
        self.op_grid.FileList.connect(self.FileList)

    def propagateDirty(self, slot, subindex, roi):
        pass

    def object_at_coordinate(self, coordinate5d):
        return self.op_grid.object_id_at(coordinate5d)


class ProgressAggregator:
    def __init__(self):
        self._progress_signal = OrderedSignal()
        self._signals: dict[OrderedSignal, float] = {}

    def __call__(self, p: OrderedSignal, value: float):
        self._signals[p] = value

        if all(v == 100 for v in self._signals.values()):
            self._progress_signal(100)
            self._signals = {}
        elif accumulated_progress := [x for x in self._signals.values() if x >= 0]:
            self._progress_signal(sum(accumulated_progress) / len(self._signals))
        elif any(x == -1 for x in self._signals.values()):
            self._progress_signal(-1)
        else:
            print(list(self._signals.items()))

    @property
    def progress_signal(self):
        return self._progress_signal

    def subscribe_to(self, signal: OrderedSignal):
        signal.subscribe(partial(self, signal))


class OpOCC(Operator):
    name = "Object Classification for Image Collections"
    category = "object classification"

    FileList = InputSlot[FileListT]()
    Embedding = InputSlot[EmbeddingTable](stype="object")
    EmbeddingModelPlugin = InputSlot[str](value="dinov2:small-reg")
    SubsetObjects = InputSlot[list[ItemId]](optional=True)
    LabelInputs = InputSlot(stype=Opaque, rtype=Index, optional=True)
    FreezePredictions = InputSlot(stype="bool", value=True)
    AdaptionParameters = InputSlot["AdaptionParametersT"](stype="object", optional=True)
    SelectEmbeddingSource = InputSlot[EmbeddingSource](stype="object")
    FreezeProjector = InputSlot(stype="bool", value=True)

    GridImage = OutputSlot()
    GridLabels = OutputSlot()

    CachedPredictions = OutputSlot[PredictionTable]()
    GridPredictions = OutputSlot()

    Eraser = OutputSlot()
    DeleteLabel = OutputSlot()
    LabelNames = OutputSlot[list[str]](stype="object")
    LabelColors = OutputSlot[list[QColor]](stype="object")
    PmapColors = OutputSlot[list[QColor]](stype="Object")
    Classifier = OutputSlot[ParallelVigraRfLazyflowClassifier]()
    ClassifierAdapted = OutputSlot[ParallelVigraRfLazyflowClassifier]()
    Umap = OutputSlot[UmapTable](stype="object")
    UmapCacheInput = InputSlot[UmapTable](stype="object", optional=True)

    AdaptedProjectorData = OutputSlot[ProjectorData](stype="object")
    AdaptedProjectorDataCacheInput = InputSlot[ProjectorData](stype="object", optional=True)
    AdaptedEmbeddings = OutputSlot["EmbeddingTable"](stype="object")
    EmbeddingCacheInput = InputSlot["EmbeddingTable"](stype="object", optional=True)
    AdaptedUmap = OutputSlot[UmapTable](stype="object")
    AdaptedUmapCacheInput = InputSlot[UmapTable](stype="object", optional=True)

    UmapForDisplay = OutputSlot[UmapTable](stype="object")

    AnnotationsTable = OutputSlot["LabelTableT"](stype="object")
    AnnotationsTableCacheInput = InputSlot["LabelTableT"](stype="object", optional=True)

    def __init__(self, graph=None, parent=None):
        super().__init__(graph=graph, parent=parent)
        self._progress_aggregator = ProgressAggregator()
        self.progress_signal = self._progress_aggregator.progress_signal
        self._got_lane = False
        self.op_grid = OpGrid(parent=self)

        self.op_grid_image = OpGridView(parent=self, FileList=self.FileList)
        self.GridImage.connect(self.op_grid_image.Output)

        # hack for now this acts as it's own cache....
        self.op_grid_labels = OpGridLabels(parent=self)

        self.op_grid_label_view = OpGridLabelImage(parent=self, LabelTable=self.op_grid_labels.LabelTable)
        self.GridLabels.connect(self.op_grid_label_view.GridLabels)

        self.op_umap = OpUmap(parent=self, Features=self.Embedding)
        self._progress_aggregator.subscribe_to(self.op_umap.progress_signal)
        self.umap_cache = OpValueCache[UmapTable](parent=self)
        self.umap_cache.Input.connect(self.op_umap.Umap)
        self.Umap.connect(self.umap_cache.Output)

        self.op_embedding_adapt = OpEmbeddingAdapt(
            parent=self,
            FileList=self.FileList,
            Input=self.op_grid_labels.LabelTable,
            AdaptionParameters=self.AdaptionParameters,
        )
        self._progress_aggregator.subscribe_to(self.op_embedding_adapt.progress_signal)
        self.projector_cache = OpValueCache[ProjectorData](parent=self)
        self.projector_cache.Input.connect(self.op_embedding_adapt.Projector)
        self.projector_cache.fixAtCurrent.connect(self.FreezeProjector)
        self.AdaptedProjectorData.connect(self.projector_cache.Output)

        self.op_embedding = OpEmbedding(
            parent=self, Embedding=self.Embedding, FineTunedProjectorData=self.projector_cache.Output
        )
        self._progress_aggregator.subscribe_to(self.op_embedding.progress_signal)

        self.embedding_cache = OpValueCache[EmbeddingTable](parent=self)
        self.embedding_cache.Input.connect(self.op_embedding.ProjectedEmbedding)

        self.AdaptedEmbeddings.connect(self.embedding_cache.Output)

        self.op_umap_adapted = OpUmap(parent=self, Features=self.embedding_cache.Output)
        self._progress_aggregator.subscribe_to(self.op_umap_adapted.progress_signal)
        self.umap_adapted_cache = OpValueCache[UmapTable](parent=self)
        self.umap_adapted_cache.Input.connect(self.op_umap_adapted.Umap)
        self.AdaptedUmap.connect(self.umap_adapted_cache.Output)

        self.op_train_original_embedding = OpObjectCollectionTrain(
            parent=self,
            Labels=self.op_grid_labels.LabelTable,
            Features=self.Embedding,
        )
        self.classifier_cache_original_embedding = OpValueCache[ParallelVigraRfLazyflowClassifier](parent=self)
        self.classifier_cache_original_embedding.Input.connect(self.op_train_original_embedding.Classifier)
        self.Classifier.connect(self.classifier_cache_original_embedding.Output)

        self.op_train_adapted_embedding = OpObjectCollectionTrain(
            parent=self,
            Labels=self.op_grid_labels.LabelTable,
            Features=self.embedding_cache.Output,
        )
        self.classifier_cache_adapted_embedding = OpValueCache[ParallelVigraRfLazyflowClassifier](parent=self)
        self.classifier_cache_adapted_embedding.Input.connect(self.op_train_adapted_embedding.Classifier)
        self.ClassifierAdapted.connect(self.classifier_cache_adapted_embedding.Output)

        self.op_select_classifier = OpSelectSubslot[ParallelVigraRfLazyflowClassifier](parent=self)
        self.op_select_classifier.Inputs.resize(2)
        self.op_select_classifier.Inputs[EmbeddingSource.Original].connect(
            self.classifier_cache_original_embedding.Output
        )
        self.op_select_classifier.Inputs[EmbeddingSource.Adapted].connect(
            self.classifier_cache_adapted_embedding.Output
        )
        self.op_select_classifier.SubslotIndex.connect(self.SelectEmbeddingSource)

        self.op_select_umap = OpSelectSubslot[UmapTable](parent=self)
        self.op_select_umap.Inputs.resize(2)
        self.op_select_umap.Inputs[EmbeddingSource.Original].connect(self.umap_cache.Output)
        self.op_select_umap.Inputs[EmbeddingSource.Adapted].connect(self.umap_adapted_cache.Output)
        self.op_select_umap.SubslotIndex.connect(self.SelectEmbeddingSource)
        self.UmapForDisplay.connect(self.op_select_umap.Output)

        self.op_select_embedding = OpSelectSubslot[EmbeddingTable](parent=self)
        self.op_select_embedding.Inputs.resize(2)
        self.op_select_embedding.Inputs[EmbeddingSource.Original].connect(self.Embedding)
        self.op_select_embedding.Inputs[EmbeddingSource.Adapted].connect(self.embedding_cache.Output)
        self.op_select_embedding.SubslotIndex.connect(self.SelectEmbeddingSource)

        self.op_predict = OpObjectCollectionPredict(
            parent=self, Features=self.op_select_embedding.Output, Classifier=self.op_select_classifier.Output
        )
        self.prediction_cache = OpValueCache[PredictionTable](parent=self)
        self.prediction_cache.fixAtCurrent.connect(self.FreezePredictions)
        self.prediction_cache.Input.connect(self.op_predict.Predictions)
        self.CachedPredictions.connect(self.prediction_cache.Output)
        self.op_predict_image = OpGridPredictionsImage(parent=self, Predictions=self.op_predict.Predictions)
        self.grid_prediction_cache = OpSlicedBlockedArrayCache(parent=self)
        self.grid_prediction_cache.fixAtCurrent.connect(self.FreezePredictions)
        self.grid_prediction_cache.Input.connect(self.op_predict_image.GridPredictions)

        self.GridPredictions.connect(self.grid_prediction_cache.Output)

        def update_grid(*args):
            print("updating grid")
            self.GridImage.disconnect()
            self.GridLabels.disconnect()
            self.GridPredictions.disconnect()
            if self.op_grid.Grid.ready():
                grid = self.op_grid.Grid.value

                self.op_grid_image.Grid.setValue(grid)
                self.op_grid_label_view.Grid.setValue(grid)
                self.op_predict_image.Grid.setValue(grid)

                self.GridImage.connect(self.op_grid_image.Output)
                self.GridLabels.connect(self.op_grid_label_view.GridLabels)
                self.GridPredictions.connect(self.grid_prediction_cache.Output)

        # for when subselection changes
        self.op_grid.Grid.notifyDirty(update_grid)
        # for the initial connect in connectLane
        self.op_grid.Grid.notifyReady(update_grid)

        self.op_grid.SubsetObjects.connect(self.SubsetObjects)
        self.op_grid.FileList.connect(self.FileList)

        self.AnnotationsTable.connect(self.op_grid_labels.LabelTable)
        self.Eraser.setValue(100)
        self.DeleteLabel.setValue(-1)
        self.LabelNames.setValue([])

        # need to be initialized
        self.LabelNames.setValue([])
        self.LabelColors.setValue([])
        self.PmapColors.setValue([])

        def _updateNumClasses(*args):
            """
            When the number of labels changes, we MUST make sure that the prediction image changes its shape (the number of channels).
            Since setupOutputs is not called for mere dirty notifications, but is called in response to setValue(),
            we use this function to call setValue().
            """
            numClasses = len(self.LabelNames.value)
            self.op_train_original_embedding.NumClasses.setValue(numClasses)
            self.op_train_adapted_embedding.NumClasses.setValue(numClasses)

        self.LabelNames.notifyDirty(_updateNumClasses)

        self.label_view_op = OpLabelView(
            parent=self, FileList=self.FileList, AnnotationsTable=self.op_grid_labels.LabelTable
        )

    def setupOutputs(self):
        axisOrder = _OUTPUT_AXIS_KEYS
        blockDimsX = {"t": (1, 1), "z": (512, 512), "y": (512, 512), "x": (1, 1), "c": (100, 100)}
        blockDimsY = {"t": (1, 1), "z": (512, 512), "y": (1, 1), "x": (512, 512), "c": (100, 100)}
        blockDimsZ = {"t": (1, 1), "z": (1, 1), "y": (512, 512), "x": (512, 512), "c": (100, 100)}
        blockShapeX = tuple(blockDimsX[k][1] for k in axisOrder)
        blockShapeY = tuple(blockDimsY[k][1] for k in axisOrder)
        blockShapeZ = tuple(blockDimsZ[k][1] for k in axisOrder)

        self.grid_prediction_cache.BlockShape.setValue((blockShapeX, blockShapeY, blockShapeZ))

    def propagateDirty(self, slot, subindex, roi):
        pass

    def object_at_coordinate(self, coordinate5d):
        return self.op_grid.object_id_at(coordinate5d)

    def prepareObjectLabels(self, coordinate) -> tuple[int, int]:
        """
        Prepare label list for updating the object label at the given coordinate

        Determines first the id of the object and then expands the length of the
        label array if necessary.
        """
        # get grid index -> map to flat index
        object_id = self.op_grid.object_id_at(coordinate)
        if object_id is None:
            raise InvalidObjectIndex

        label_table = self.op_grid_labels.LabelTable.value
        if object_id in label_table:
            old_label = label_table[object_id].label
        else:
            old_label = 0
        dirty_key = object_id
        return old_label, dirty_key

    def _setInSlot(self, slot, subindex: int, roi: "Roi", value: Any):
        # For annotations from UI
        if slot == self.LabelInputs:
            self.op_grid_labels.Input[roi._pslice] = value

        if slot == self.AnnotationsTableCacheInput:
            self.op_grid_labels.forceValue(value)

        if slot == self.UmapCacheInput:
            self.umap_cache.forceValue(value)

        if slot == self.AdaptedUmapCacheInput:
            self.umap_adapted_cache.forceValue(value)

        if slot == self.EmbeddingCacheInput:
            self.embedding_cache.forceValue(value)

        if slot == self.AdaptedProjectorDataCacheInput:
            self.projector_cache.forceValue(value)

    def addLane(self, laneIndex: int):
        assert laneIndex == 0
        assert not self._got_lane

        self._got_lane = True

    def removeLane(self, laneIndex: int, finalLength: int):
        assert laneIndex == 0
        assert finalLength == 0
        assert self._got_lane

        self._got_lane = False

    def getLane(self, laneIndex: int):
        assert laneIndex == 0
        assert self._got_lane
        return self

    def cancellation_token(self, token):
        self.op_embedding_adapt.cancellation_token = token

    def clearLabel(self, label: int):
        # set this label to 0 in the label inputs
        assert self._got_lane
        annotation_label = label + 1
        labels = {k: v for k, v in self.op_grid_labels.LabelTable.value.items() if v.label != annotation_label}
        self.op_grid_labels.forceValue(labels)

    def removeLabel(self, label: int):
        assert self._got_lane
        print(f"removing label {label}")
        annotation_label = label + 1

        def _maybe_decr_label(label_row: LabelRow):
            if label_row.label < annotation_label:
                return label_row

            return LabelRow(label_row.id, label_row.label - 1)

        new_labels = {
            k: _maybe_decr_label(v)
            for k, v in self.op_grid_labels.LabelTable.value.items()
            if v.label != annotation_label
        }

        self.op_grid_labels.forceValue(new_labels)
