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
#          http://ilastik.org/license.html
###############################################################################
from typing import Any, Callable, Optional, TypeAlias, Union

from lazyflow.cancel_token import CancellationToken
import numpy.typing as npt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from ilastik.applets.objectClassificationCollection.types import LabelRow, LabelTable
from lazyflow.operators.ioOperators.types import FileList
from lazyflow.utility.grid import _OUTPUT_AXIS_KEYS

DeviceLikeType: TypeAlias = Union[str, torch.device, int]


class DinoV2Backbone(nn.Module):
    def __init__(self, model_name: str = "dinov2_vits14_reg"):
        super().__init__()
        print(f"Loading DINOv2 model: {model_name}")
        self.model: nn.Module = torch.hub.load("facebookresearch/dinov2:main", model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            return self.model(x)


class Projector(nn.Module):
    """
    Maps a DINOv2 embedding to:
      - a hidden representation (used for classification / saved features),
      - a normalized projection (used by the three adaptation losses).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        hidden = self.act(self.bn(self.layer1(x)))
        projected = F.normalize(self.layer2(hidden), dim=-1)
        return hidden, projected


class PrototypeAligner:
    """
    Keeps one EMA prototype per class, built from the labeled images, and pulls
    the confidently pseudo-labeled unlabeled images towards them.
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        momentum: float,
        threshold: float,
        temperature: float,
        device: DeviceLikeType,
    ):
        self.num_classes = num_classes
        self.momentum = momentum
        self.threshold = threshold
        self.temperature = temperature
        self.device = device
        self.prototypes = torch.zeros(num_classes, feat_dim, device=device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, z: torch.Tensor, labels: npt.NDArray[np.integer]):
        z = F.normalize(z, dim=-1)
        for c in range(self.num_classes):
            mask = labels == c
            if not mask.any():
                continue
            centroid = F.normalize(z[mask].mean(dim=0), dim=0)
            if self.initialized[c]:
                self.prototypes[c] = F.normalize(
                    self.momentum * self.prototypes[c] + (1.0 - self.momentum) * centroid, dim=0
                )
            else:
                self.prototypes[c] = centroid
                self.initialized[c] = True

    def loss(self, z_unlabeled: torch.Tensor):
        """
        Returns (loss, number of confidently pseudo-labeled views).
        Stays at zero until every class has a prototype.
        """
        if not self.initialized.all() or z_unlabeled.shape[0] == 0:
            return torch.tensor(0.0, device=self.device), 0
        sims = torch.mm(F.normalize(z_unlabeled, dim=-1), self.prototypes.T) / self.temperature
        max_probs, pseudo_labels = F.softmax(sims, dim=-1).max(dim=-1)
        confident = max_probs > self.threshold
        n_confident = int(confident.sum().item())
        if n_confident == 0:
            return torch.tensor(0.0, device=self.device), 0
        return F.cross_entropy(sims[confident], pseudo_labels[confident]), n_confident


class TwoViewDataset(torch.utils.data.Dataset):
    """Returns two augmented views of an image. Unlabeled items carry label -1.

    Args:
        file_list: references to data used in fine tuning
        labels: sparse label list of only labeled data - may contain zeros for data
          that has been previously labeled.
        transform:
    """

    def __init__(
        self,
        file_list: FileList,
        labels: LabelTable,
        transform: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    ):

        self._file_list = file_list
        self._indices = list(self._file_list.keys())
        self._labels = labels
        self.transform = transform

    def __len__(self):
        return len(self._file_list)

    def __getitem__(self, idx: int) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], int]:
        f_index = self._indices[idx]
        item = self._file_list[f_index]
        data = self._ensure_rgb(item.accessor(), _OUTPUT_AXIS_KEYS)
        label = self._labels.get(f_index)
        if label is None or label.label == 0:
            label_number = -1
        else:
            label_number = label.label
        return self.transform(data), self.transform(data), label_number

    @staticmethod
    def _ensure_rgb(image_array: npt.NDArray[np.floating], axistags: str):
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


def build_mixed_items(items: FileList, labels: LabelTable, labeled_fraction: float) -> tuple[FileList, LabelTable]:
    """
    Repeats the labeled items so that they make up roughly `labeled_fraction`
    of every adaptation batch, then appends the unlabeled ones.

    TODO: probably also take in the number of unlabeled items one wants to have
          or some total number. - check kostas code

    """
    assert 0.0 < labeled_fraction < 1.0
    assert len(items) > 0
    assert len(labels) > 0

    labeled_items = [(items[idx], labels[idx]) for idx in labels if labels[idx].label != 0]
    unlabeled_items = [
        (items[idx], LabelRow(id=idx, label=-1)) for idx in items if idx not in labels or labels[idx].label == 0
    ]

    target = max(
        int(round(len(unlabeled_items) * labeled_fraction / (1.0 - labeled_fraction))),
        len(labeled_items),
    )
    repeats = target // len(labeled_items) + 1
    all_items = (labeled_items * repeats)[:target] + unlabeled_items
    return {x[0].id: x[0] for x in all_items}, {x[1].id: x[1] for x in all_items}


def nt_xent_loss(z: torch.Tensor, temperature: float):
    """
    Self-supervised loss. z holds the two views of every image interleaved:
    [img0_view1, img0_view2, img1_view1, ...], all L2-normalized.
    """
    num_pairs = z.shape[0] // 2
    sim = torch.mm(z, z.T) / temperature
    sim.masked_fill_(torch.eye(2 * num_pairs, dtype=torch.bool, device=z.device), -1e9)
    pos_indices = torch.arange(2 * num_pairs, device=z.device)
    pos_indices[0::2] += 1
    pos_indices[1::2] -= 1
    pos_sim = sim[torch.arange(2 * num_pairs, device=z.device), pos_indices]
    return (-pos_sim + torch.logsumexp(sim, dim=1)).mean()


def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float):
    """
    Supervised loss. Every view of the same class is a positive of every other.
    """
    if z.shape[0] <= 1:
        return torch.tensor(0.0, device=z.device)
    sim = torch.mm(z, z.T) / temperature
    self_mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, -1e9)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    if not pos_mask.any():
        return torch.tensor(0.0, device=z.device)
    log_sum_exp = torch.logsumexp(sim, dim=1)
    pos_counts = pos_mask.sum(dim=1).clamp(min=1).float()
    return (-(pos_mask.float() * (sim - log_sum_exp.unsqueeze(1))).sum(dim=1) / pos_counts).mean()


class RandomRotation90:
    def __call__(self, img):
        k = torch.randint(1, 4, ()).item()  # 1, 2, or 3
        return torch.rot90(img, k=k, dims=(-2, -1))


class GaussianNoise:
    def __init__(self, std: float = 0.02, p: float = 0.3):
        self._std = std
        self._p = p

    def __call__(self, tensor: torch.Tensor):
        if torch.rand(1).item() < self._p:
            return tensor + torch.randn_like(tensor) * self._std
        return tensor


def build_transform(is_train: bool):
    """Strong augmentations for the two views, plain resize/crop for evaluation."""
    if is_train:
        return T.Compose(
            [
                T.RandomResizedCrop(
                    224, scale=(0.6, 1.0), ratio=(0.85, 1.15), interpolation=T.InterpolationMode.BICUBIC
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                RandomRotation90(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                GaussianNoise(std=0.02, p=0.3),
            ]
        )
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def adapt_features(
    items: FileList,
    item_labels: LabelTable,
    num_classes: int,
    backbone: nn.Module,
    projector: nn.Module,
    device: DeviceLikeType,
    labeled_fraction: float,
    cancellation_token: CancellationToken,
    batch_size: int = 64,
    num_workers: int = 4,
    adapt_epochs: int = 15,
    temperature: float = 0.07,
    proj_out_dim: int = 128,
    proto_momentum: float = 0.99,
    proto_threshold: float = 0.95,
    proto_temperature: float = 0.1,
    lambda_sup: float = 1.0,
    lambda_proto: float = 0.5,
    adapt_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    progress_callback: Optional[Callable[[float], None]] = None,
    message_callback: Optional[Callable[[str], None]] = None,
):
    def nop(_arg: float | str):
        pass

    message_callback = message_callback or nop
    progress_callback = progress_callback or nop
    mixed_items, mixed_labels = build_mixed_items(items, item_labels, labeled_fraction)
    loader = torch.utils.data.DataLoader(
        TwoViewDataset(mixed_items, mixed_labels, build_transform(is_train=True)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Hack: no mp for now, cannot pickle data accessor...
        drop_last=len(mixed_items) > batch_size,
    )
    message_callback(f"Adaptation set: {len(mixed_items)} images " f"({len(item_labels)} labeled repeated)")
    progress_callback(-1)
    optimizer = optim.AdamW(projector.parameters(), lr=adapt_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(adapt_epochs * len(loader), 1), eta_min=adapt_lr * 0.01
    )
    prototypes = PrototypeAligner(
        num_classes=num_classes,
        feat_dim=proj_out_dim,
        momentum=proto_momentum,
        threshold=proto_threshold,
        temperature=proto_temperature,
        device=device,
    )

    history: list[dict[str, Any]] = []
    projector.train()
    progress_callback(0.0)
    for epoch in range(adapt_epochs):
        if cancellation_token.cancelled:
            message_callback(f"Cancellation requested by user after epoch {epoch - 1}")
            return history
        totals = {"total": 0.0, "self": 0.0, "sup": 0.0, "proto": 0.0}
        pseudo_labeled, batches = 0, 0

        for view1, view2, labels in loader:
            view1, view2 = view1.to(device), view2.to(device)
            labels = labels.to(device)

            _, projected1 = projector(backbone(view1))
            _, projected2 = projector(backbone(view2))

            # Self-supervised: the two views of an image are the only positives
            interleaved = torch.stack([projected1, projected2], dim=1).reshape(2 * view1.shape[0], -1)
            loss_self = nt_xent_loss(interleaved, temperature)

            # Supervised: same-class views attract, and they update the prototypes
            labeled_mask = labels >= 0
            if labeled_mask.any():
                z_labeled = torch.cat([projected1[labeled_mask], projected2[labeled_mask]], dim=0)
                labeled_targets = torch.cat([labels[labeled_mask], labels[labeled_mask]], dim=0)
                loss_sup = supervised_contrastive_loss(z_labeled, labeled_targets, temperature)
                prototypes.update(z_labeled.detach(), labeled_targets)
            else:
                loss_sup = torch.tensor(0.0, device=device)

            # Pseudo-label: confident unlabeled views are pulled to their prototype
            unlabeled_mask = labels < 0
            if unlabeled_mask.any() and lambda_proto > 0:
                z_unlabeled = torch.cat([projected1[unlabeled_mask], projected2[unlabeled_mask]], dim=0)
                loss_proto, n_confident = prototypes.loss(z_unlabeled)
            else:
                loss_proto, n_confident = torch.tensor(0.0, device=device), 0

            loss = loss_self + lambda_sup * loss_sup + lambda_proto * loss_proto
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            totals["total"] += loss.item()
            totals["self"] += loss_self.item()
            totals["sup"] += loss_sup.item()
            totals["proto"] += loss_proto.item()
            pseudo_labeled += n_confident
            batches += 1

        history.append(
            {
                "epoch": epoch + 1,
                "loss_total": totals["total"] / max(batches, 1),
                "loss_self": totals["self"] / max(batches, 1),
                "loss_sup": totals["sup"] / max(batches, 1),
                "loss_proto": totals["proto"] / max(batches, 1),
                "pseudo_labeled": pseudo_labeled,
            }
        )
        progress_callback((epoch + 1) / adapt_epochs * 100)
        message_callback(
            f"[Adapt {epoch+1}/{adapt_epochs}] "
            f"Total: {history[-1]['loss_total']:.4f} | Self: {history[-1]['loss_self']:.4f} | "
            f"Sup: {history[-1]['loss_sup']:.4f} | Proto: {history[-1]['loss_proto']:.4f} | "
            f"Pseudo-labeled: {pseudo_labeled}"
        )

    projector.eval()
    progress_callback(100.0)
    return history
