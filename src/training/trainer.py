"""Training loop with AMP, scheduling, class weighting and early stopping."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..utils.config import Config


def _build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    name = cfg.training.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(
            params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.training.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=cfg.training.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {cfg.training.optimizer!r}")


def _build_scheduler(optimizer, cfg: Config):
    name = cfg.training.scheduler.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {cfg.training.scheduler!r}")


class Trainer:
    """Encapsulates the full fit/evaluate lifecycle for one model."""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        class_names: list[str],
        class_weights: torch.Tensor | None = None,
        device: str | None = None,
    ) -> None:
        self.cfg = config
        self.class_names = class_names
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)

        weight = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=config.training.label_smoothing
        )
        self.optimizer = _build_optimizer(self.model, config)
        self.scheduler = _build_scheduler(self.optimizer, config)

        self.use_amp = config.training.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.ckpt_dir = Path(config.output.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.output.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    # ------------------------------------------------------------- one epoch
    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss, correct, seen = 0.0, 0, 0
        desc = "train" if train else "val"

        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            seen += images.size(0)

        return total_loss / max(seen, 1), correct / max(seen, 1)

    # -------------------------------------------------------------- fit
    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> dict[str, list[float]]:
        best_val_acc = -1.0
        epochs_no_improve = 0
        patience = self.cfg.training.early_stopping_patience

        for epoch in range(1, self.cfg.training.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)
            if self.scheduler is not None:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(lr)

            print(
                f"Epoch {epoch:02d}/{self.cfg.training.epochs} "
                f"| train loss {train_loss:.4f} acc {train_acc:.4f} "
                f"| val loss {val_loss:.4f} acc {val_acc:.4f} "
                f"| lr {lr:.2e} | {time.time() - t0:.0f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                self.save_checkpoint("best.pth", epoch=epoch, val_acc=val_acc)
            else:
                epochs_no_improve += 1
                if patience and epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (best val acc {best_val_acc:.4f}).")
                    break

        self.save_checkpoint("last.pth", epoch=epoch, val_acc=val_acc)
        with open(self.log_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        return self.history

    # ----------------------------------------------------------- evaluate
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_true, y_pred) over a loader."""
        self.model.eval()
        all_true, all_pred = [], []
        for images, targets in tqdm(loader, desc="predict", leave=False):
            images = images.to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
            all_pred.append(logits.argmax(1).cpu().numpy())
            all_true.append(targets.numpy())
        return np.concatenate(all_true), np.concatenate(all_pred)

    # ----------------------------------------------------------- checkpoint
    def save_checkpoint(self, name: str, **extra: Any) -> Path:
        path = self.ckpt_dir / name
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "class_names": self.class_names,
                "config": self.cfg.to_dict(),
                **extra,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
