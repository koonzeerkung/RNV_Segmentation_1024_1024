from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Literal

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SOURCE_SIZE = 1024
INPUT_SIZE = SOURCE_SIZE
PATCH_SIZE = INPUT_SIZE
PATCHES_PER_IMAGE = 1
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
AFFINE_ROTATE_LIMIT = (-15, 15)
AFFINE_SCALE_LIMIT = (0.9, 1.1)
AFFINE_TRANSLATE_LIMIT = (-0.05, 0.05)
AFFINE_SHEAR_LIMIT = (-5, 5)

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SliceSample:
    image_path: Path
    mask_path: Path
    dataset: str
    sample_id: int
    patch_id: int
    crop_box: tuple[int, int, int, int]
    split: SplitName


def normalize_dataset_name(dataset: str) -> str:
    """Return the project dataset folder name from `tu`, `drac`, or full names."""

    normalized = dataset.lower().strip()
    aliases = {
        "tu": "dataset_tu",
        "dataset_tu": "dataset_tu",
        "drac": "dataset_drac",
        "dataset_drac": "dataset_drac",
    }
    if normalized not in aliases:
        raise ValueError("dataset must be one of: 'tu', 'drac', 'dataset_tu', 'dataset_drac'")
    return aliases[normalized]


def get_kfold_chunks(dataset: str) -> list[list[int]]:
    dataset = normalize_dataset_name(dataset)
    if dataset == "dataset_tu":
        return [
            list(range(1, 7)),
            list(range(7, 13)),
            list(range(13, 18)),
            list(range(18, 23)),
            list(range(23, 28)),
        ]
    if dataset == "dataset_drac":
        return [
            list(range(1, 10)),
            list(range(10, 19)),
            list(range(19, 27)),
            list(range(27, 35)),
            list(range(35, 43)),
        ]
    raise ValueError("dataset must be 'tu' or 'drac'")


def _is_augment_enabled(augment: str | bool) -> bool:
    if isinstance(augment, bool):
        return augment

    value = augment.lower().strip()
    if value in {"yes", "y", "true", "1"}:
        return True
    if value in {"no", "n", "false", "0"}:
        return False
    raise ValueError("augment must be 'yes', 'no', True, or False")


def find_image_file(directory: Path, sample_id: int) -> Path:
    candidate_stems = (
        str(sample_id),
        f"nv ({sample_id})",
        f"gt ({sample_id})",
    )
    for stem in candidate_stems:
        for extension in IMAGE_EXTENSIONS:
            path = directory / f"{stem}{extension}"
            if path.exists():
                return path

    id_pattern = re.compile(rf"(?<!\d){sample_id}(?!\d)")
    for extension in IMAGE_EXTENSIONS:
        matches = sorted(
            path
            for path in directory.glob(f"*{extension}")
            if id_pattern.search(path.stem)
        )
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No image file found for sample {sample_id} in {directory}")


def _resolve_dataset_dir(base_dir: str | Path, dataset_name: str) -> Path:
    dataset_token = dataset_name.replace("dataset_", "")
    candidates = [
        Path(base_dir) / dataset_name,
        Path(base_dir) / dataset_token.upper(),
        Path(base_dir) / dataset_token,
        Path(base_dir) / "dataset" / dataset_token.upper(),
        Path(base_dir) / "dataset" / dataset_token,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _resolve_data_subdir(dataset_dir: Path, subdir_name: str) -> Path:
    for candidate in (
        dataset_dir / subdir_name,
        dataset_dir / subdir_name.upper(),
        dataset_dir / subdir_name.capitalize(),
    ):
        if candidate.is_dir():
            return candidate
    return dataset_dir / subdir_name


def resolve_dataset_layout(dataset: str, base_dir: str | Path = ".") -> tuple[str, Path, Path, Path]:
    dataset_name = normalize_dataset_name(dataset)
    dataset_dir = _resolve_dataset_dir(base_dir, dataset_name)
    raw_dir = _resolve_data_subdir(dataset_dir, "raw")
    gt_dir = _resolve_data_subdir(dataset_dir, "gt")
    return dataset_name, dataset_dir, raw_dir, gt_dir


def _split_ids(dataset: str, fold: int) -> dict[SplitName, list[int]]:
    chunks = get_kfold_chunks(dataset)
    if not 1 <= fold <= len(chunks):
        raise ValueError(f"fold must be between 1 and {len(chunks)}")

    test_index = fold - 1
    validation_index = fold % len(chunks)

    split_ids: dict[SplitName, list[int]] = {
        "test": chunks[test_index],
        "validation": chunks[validation_index],
        "train": [],
    }
    for index, chunk in enumerate(chunks):
        if index not in {test_index, validation_index}:
            split_ids["train"].extend(chunk)
    return split_ids


def _crop_boxes() -> list[tuple[int, int, int, int]]:
    return [
        (0, 0, SOURCE_SIZE, SOURCE_SIZE),
    ]


def _to_snake_case(value: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value.lower()


def _train_augmentation_transforms() -> list[A.BasicTransform]:
    return [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=AFFINE_SCALE_LIMIT,
            translate_percent=AFFINE_TRANSLATE_LIMIT,
            rotate=AFFINE_ROTATE_LIMIT,
            shear=AFFINE_SHEAR_LIMIT,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_REFLECT_101,
            fill=0,
            fill_mask=0,
            p=0.75,
        ),
        A.CLAHE(
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_grid_size=CLAHE_TILE_GRID_SIZE,
            p=0.5,
        ),
    ]


def build_train_augmentation(with_replay: bool = False) -> A.Compose | A.ReplayCompose:
    """Build the training augmentation pipeline."""

    transforms = _train_augmentation_transforms()
    if with_replay:
        return A.ReplayCompose(transforms)
    return A.Compose(transforms)


def build_samples(dataset: str, fold: int, base_dir: str | Path = ".") -> dict[SplitName, list[SliceSample]]:
    dataset_name, _, raw_dir, gt_dir = resolve_dataset_layout(dataset, base_dir)

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw directory: {raw_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing gt directory: {gt_dir}")

    samples: dict[SplitName, list[SliceSample]] = {"train": [], "validation": [], "test": []}
    for split, sample_ids in _split_ids(dataset_name, fold).items():
        for sample_id in sample_ids:
            image_path = find_image_file(raw_dir, sample_id)
            mask_path = find_image_file(gt_dir, sample_id)
            for patch_id, crop_box in enumerate(_crop_boxes()):
                samples[split].append(
                    SliceSample(
                        image_path=image_path,
                        mask_path=mask_path,
                        dataset=dataset_name,
                        sample_id=sample_id,
                        patch_id=patch_id,
                        crop_box=crop_box,
                        split=split,
                    )
                )
    return samples


def _mask_crop_contains_answer(
    mask_path: Path,
    crop_box: tuple[int, int, int, int],
    mask_threshold: float = 0.5,
) -> bool:
    with Image.open(mask_path) as mask:
        mask_image = mask.convert("L")

    if mask_image.size != (SOURCE_SIZE, SOURCE_SIZE):
        raise ValueError(f"Expected {SOURCE_SIZE}x{SOURCE_SIZE} mask, got {mask_image.size}: {mask_path}")

    mask_crop = mask_image.crop(crop_box)
    if mask_crop.size != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"Expected {INPUT_SIZE}x{INPUT_SIZE} mask, got {mask_crop.size}: {mask_path}")

    mask_array = np.asarray(mask_crop, dtype=np.uint8).astype(np.float32) / 255.0
    return bool(np.any(mask_array >= mask_threshold))


def filter_samples_with_answers(
    samples: list[SliceSample],
    mask_threshold: float = 0.5,
) -> list[SliceSample]:
    """Keep only samples whose ground-truth mask contains foreground."""

    return [
        sample
        for sample in samples
        if _mask_crop_contains_answer(sample.mask_path, sample.crop_box, mask_threshold=mask_threshold)
    ]


class KFoldSliceDataset(Dataset):
    """Dataset that loads grayscale 1024x1024 raw/gt images from one k-fold split."""

    def __init__(
        self,
        samples: list[SliceSample],
        augment: str | bool = "no",
        aug_multiplier: int = 1,
        mask_threshold: float = 0.5,
        include_metadata: bool = True,
    ) -> None:
        if not samples:
            raise ValueError("KFoldSliceDataset requires at least one sample")

        self.samples = samples
        self.augment_enabled = _is_augment_enabled(augment)
        self.aug_multiplier = int(aug_multiplier) if self.augment_enabled else 1
        self.mask_threshold = mask_threshold
        self.include_metadata = include_metadata
        self.train_augmentation = (
            build_train_augmentation(with_replay=include_metadata) if self.augment_enabled else None
        )

        if self.aug_multiplier < 1:
            raise ValueError("aug_multiplier must be at least 1")

    def __len__(self) -> int:
        return len(self.samples) * self.aug_multiplier

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index // self.aug_multiplier]
        augment_index = index % self.aug_multiplier

        with Image.open(sample.image_path) as image_handle:
            image = image_handle.convert("L")
        with Image.open(sample.mask_path) as mask_handle:
            mask = mask_handle.convert("L")

        if image.size != (SOURCE_SIZE, SOURCE_SIZE):
            raise ValueError(f"Expected {INPUT_SIZE}x{INPUT_SIZE} image, got {image.size}: {sample.image_path}")
        if mask.size != (SOURCE_SIZE, SOURCE_SIZE):
            raise ValueError(f"Expected {INPUT_SIZE}x{INPUT_SIZE} mask, got {mask.size}: {sample.mask_path}")

        image = image.crop(sample.crop_box)
        mask = mask.crop(sample.crop_box)
        if image.size != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Expected {INPUT_SIZE}x{INPUT_SIZE} image crop, got {image.size}: {sample.image_path}")
        if mask.size != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Expected {INPUT_SIZE}x{INPUT_SIZE} mask crop, got {mask.size}: {sample.mask_path}")

        image_array = np.asarray(image, dtype=np.uint8)
        mask_array = np.asarray(mask, dtype=np.uint8)

        if self.augment_enabled and augment_index:
            image_array, mask_array, augmentation = self._apply_augmentation(image_array, mask_array)
        else:
            augmentation = "original"

        item: dict[str, object] = {
            "image": self._image_to_tensor(image_array),
            "mask": self._mask_to_tensor(mask_array),
        }
        if not self.include_metadata:
            return item

        item.update({
            "dataset": sample.dataset,
            "split": sample.split,
            "sample_id": sample.sample_id,
            "patch_id": sample.patch_id,
            "augment_index": augment_index,
            "augmentation": augmentation,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
        })
        return item

    def _apply_augmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if self.train_augmentation is None:
            return image, mask, "original"

        transformed = self.train_augmentation(image=image, mask=mask)
        replay = transformed.get("replay", {}) if self.include_metadata else {}
        augmentation = self._summarize_replay(replay) if self.include_metadata else "augmented"
        return transformed["image"], transformed["mask"], augmentation

    def _summarize_replay(self, replay: dict[str, object]) -> str:
        transforms = replay.get("transforms", [])
        applied: list[str] = []
        for transform in transforms:
            if not isinstance(transform, dict) or not transform.get("applied"):
                continue
            name = str(transform.get("__class_fullname__", "augmentation")).split(".")[-1]
            applied.append(_to_snake_case(name))
        return "+".join(applied) if applied else "none_applied"

    def _image_to_tensor(self, image_array: np.ndarray) -> torch.Tensor:
        image_array = image_array.astype(np.float32) / 255.0
        return torch.from_numpy(image_array).unsqueeze(0).contiguous()

    def _mask_to_tensor(self, mask_array: np.ndarray) -> torch.Tensor:
        mask_array = mask_array.astype(np.float32) / 255.0
        mask_array = (mask_array >= self.mask_threshold).astype(np.float32)
        return torch.from_numpy(mask_array).unsqueeze(0).contiguous()


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> DataLoader:
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def create_dataloaders(
    dataset: str,
    augment: str | bool,
    aug_multiplier: int,
    fold: int,
    batch_size: int = 8,
    base_dir: str | Path = ".",
    num_workers: int = 0,
    include_metadata: bool = True,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int | None = 2,
) -> dict[SplitName, DataLoader]:
    """
    Create train, validation, and test DataLoaders for the requested dataset/fold.

    `fold` is 1-based. The selected fold is test, the next fold is validation,
    and the remaining folds are training. Validation wraps after fold 5.
    """

    split_samples = build_samples(dataset=dataset, fold=fold, base_dir=base_dir)
    train_samples = filter_samples_with_answers(split_samples["train"])
    train_dataset = KFoldSliceDataset(
        train_samples,
        augment=augment,
        aug_multiplier=aug_multiplier,
        include_metadata=include_metadata,
    )
    validation_dataset = KFoldSliceDataset(
        split_samples["validation"],
        augment="no",
        aug_multiplier=1,
        include_metadata=include_metadata,
    )
    test_dataset = KFoldSliceDataset(
        split_samples["test"],
        augment="no",
        aug_multiplier=1,
        include_metadata=include_metadata,
    )

    return {
        "train": _make_loader(
            train_dataset, batch_size, True, num_workers, pin_memory, persistent_workers, prefetch_factor
        ),
        "validation": _make_loader(
            validation_dataset, batch_size, False, num_workers, pin_memory, persistent_workers, prefetch_factor
        ),
        "test": _make_loader(
            test_dataset, batch_size, False, num_workers, pin_memory, persistent_workers, prefetch_factor
        ),
    }


def get_dataloader(
    dataset: str,
    augment: str | bool,
    aug_multiplier: int,
    fold: int,
    batch_size: int = 8,
    base_dir: str | Path = ".",
    num_workers: int = 0,
    include_metadata: bool = True,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int | None = 2,
) -> dict[SplitName, DataLoader]:
    """Backward-compatible alias for `create_dataloaders`."""

    return create_dataloaders(
        dataset=dataset,
        augment=augment,
        aug_multiplier=aug_multiplier,
        fold=fold,
        batch_size=batch_size,
        base_dir=base_dir,
        num_workers=num_workers,
        include_metadata=include_metadata,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


__all__ = [
    "KFoldSliceDataset",
    "INPUT_SIZE",
    "PATCHES_PER_IMAGE",
    "PATCH_SIZE",
    "SliceSample",
    "SOURCE_SIZE",
    "build_train_augmentation",
    "build_samples",
    "create_dataloaders",
    "filter_samples_with_answers",
    "find_image_file",
    "get_dataloader",
    "get_kfold_chunks",
    "normalize_dataset_name",
    "resolve_dataset_layout",
]
