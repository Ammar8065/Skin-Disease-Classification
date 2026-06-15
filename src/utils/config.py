"""Typed, YAML-backed configuration.

The config is a set of nested dataclasses so that attribute access is checked
and IDE-completed, while still being loadable from / dumpable to plain YAML.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_dir: str = "data/skin-diseases"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    val_split: float = 0.15
    test_split: float = 0.10


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.3
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    label_smoothing: float = 0.1
    use_class_weights: bool = True
    early_stopping_patience: int = 6
    mixed_precision: bool = True


@dataclass
class OutputConfig:
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build a Config from a (possibly partial) nested dict."""
        section_types = {f.name: f.type for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in section_types:
                raise KeyError(f"Unknown config section: {key!r}")
            if isinstance(value, dict):
                section_cls = {
                    "data": DataConfig,
                    "model": ModelConfig,
                    "training": TrainingConfig,
                    "output": OutputConfig,
                }[key]
                kwargs[key] = section_cls(**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def load_config(path: str | Path) -> Config:
    """Load a Config from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return Config.from_dict(raw)
