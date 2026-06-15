"""Dataset discovery and DataLoader construction.

The data module is deliberately tolerant of how the downloaded dataset is laid
out on disk, because public Kaggle mirrors of the 22-class skin disease set
ship in a few different shapes:

  * ``<root>/train/<class>/*.jpg`` and ``<root>/test/<class>/*.jpg``
  * ``<root>/<class>/*.jpg`` (single split — we create val/test ourselves)

Class names are taken from the discovered folder names so training never
depends on the exact spelling in ``labels.py`` (which is the canonical,
human-readable reference list).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from .transforms import build_transforms

_TRAIN_DIR_NAMES = ("train", "training")
_VAL_DIR_NAMES = ("val", "valid", "validation")
_TEST_DIR_NAMES = ("test", "testing")


def _find_subdir(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _resolve_image_root(data_dir: Path) -> Path:
    """Descend through single-child wrapper folders.

    Kaggle archives often unzip into ``<root>/<archive-name>/...``. If the given
    directory has exactly one subdirectory and no images, step into it.
    """
    current = data_dir
    for _ in range(5):  # guard against pathological nesting
        subdirs = [p for p in current.iterdir() if p.is_dir()]
        has_split = any(_find_subdir(current, n) for n in
                        (_TRAIN_DIR_NAMES, _VAL_DIR_NAMES, _TEST_DIR_NAMES))
        if has_split or len(subdirs) != 1:
            return current
        current = subdirs[0]
    return current


class SkinDiseaseDataModule:
    """Builds train/val/test DataLoaders from a dataset directory."""

    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 2,
        val_split: float = 0.15,
        test_split: float = 0.10,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.class_names: list[str] = []
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        # Integer labels for the training split (used for class weighting).
        self._train_targets: np.ndarray | None = None

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir}. "
                "Download/extract the dataset first (see the Colab notebook)."
            )

        root = _resolve_image_root(self.data_dir)
        train_dir = _find_subdir(root, _TRAIN_DIR_NAMES)

        train_tf = build_transforms(self.image_size, train=True)
        eval_tf = build_transforms(self.image_size, train=False)

        if train_dir is not None:
            self._setup_from_predefined_splits(root, train_dir, train_tf, eval_tf)
        else:
            self._setup_from_single_folder(root, train_tf, eval_tf)

    def _setup_from_predefined_splits(self, root, train_dir, train_tf, eval_tf):
        full_train = datasets.ImageFolder(str(train_dir), transform=train_tf)
        self.class_names = full_train.classes

        test_dir = _find_subdir(root, _TEST_DIR_NAMES)
        val_dir = _find_subdir(root, _VAL_DIR_NAMES)

        if val_dir is not None:
            self.train_dataset = full_train
            self.val_dataset = datasets.ImageFolder(str(val_dir), transform=eval_tf)
        else:
            # Carve a validation split out of train.
            train_idx, val_idx = self._stratified_split(
                full_train.targets, self.val_split
            )
            eval_view = datasets.ImageFolder(str(train_dir), transform=eval_tf)
            self.train_dataset = Subset(full_train, train_idx)
            self.val_dataset = Subset(eval_view, val_idx)
            self._train_targets = np.array(full_train.targets)[train_idx]

        if test_dir is not None:
            self.test_dataset = datasets.ImageFolder(str(test_dir), transform=eval_tf)
        else:
            self.test_dataset = self.val_dataset

        if self._train_targets is None:
            self._train_targets = np.array(
                _targets_of(self.train_dataset)
            )

    def _setup_from_single_folder(self, root, train_tf, eval_tf):
        base = datasets.ImageFolder(str(root), transform=train_tf)
        eval_view = datasets.ImageFolder(str(root), transform=eval_tf)
        self.class_names = base.classes
        targets = np.array(base.targets)

        train_idx, holdout_idx = self._stratified_split(
            targets, self.val_split + self.test_split
        )
        # Split the holdout into val/test proportionally.
        rel_test = self.test_split / max(self.val_split + self.test_split, 1e-9)
        val_idx, test_idx = self._stratified_split(
            targets[holdout_idx], rel_test
        )
        val_idx = holdout_idx[val_idx]
        test_idx = holdout_idx[test_idx]

        self.train_dataset = Subset(base, train_idx)
        self.val_dataset = Subset(eval_view, val_idx)
        self.test_dataset = Subset(eval_view, test_idx)
        self._train_targets = targets[train_idx]

    # ------------------------------------------------------------- splitting
    def _stratified_split(
        self, targets, holdout_fraction: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (keep_idx, holdout_idx), stratified by class."""
        rng = np.random.default_rng(self.seed)
        targets = np.asarray(targets)
        keep, holdout = [], []
        for cls in np.unique(targets):
            cls_idx = np.where(targets == cls)[0]
            rng.shuffle(cls_idx)
            n_holdout = max(1, int(round(len(cls_idx) * holdout_fraction)))
            holdout.extend(cls_idx[:n_holdout])
            keep.extend(cls_idx[n_holdout:])
        return np.array(keep), np.array(holdout)

    # ----------------------------------------------------------- properties
    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights from the training split."""
        if self._train_targets is None:
            raise RuntimeError("Call setup() before requesting class weights.")
        counts = np.bincount(self._train_targets, minlength=self.num_classes)
        counts = np.clip(counts, 1, None)
        weights = counts.sum() / (self.num_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)

    # ------------------------------------------------------------ loaders
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def _targets_of(dataset: Dataset) -> list[int]:
    """Extract integer targets from an ImageFolder or a Subset thereof."""
    if isinstance(dataset, Subset):
        base_targets = dataset.dataset.targets  # type: ignore[attr-defined]
        return [base_targets[i] for i in dataset.indices]
    return list(dataset.targets)  # type: ignore[attr-defined]
