"""Command-line training entry point.

Usage:
    python -m scripts.train --config configs/default.yaml
    python -m scripts.train --config configs/default.yaml --data-dir /path/to/data --epochs 30

Run from the project root so that ``src`` is importable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import SkinDiseaseDataModule  # noqa: E402
from src.models import build_model  # noqa: E402
from src.training import Trainer, classification_metrics, plot_confusion_matrix  # noqa: E402
from src.utils import load_config, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the skin disease classifier.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data-dir", default=None, help="Override config data.data_dir")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--backbone", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.batch_size:
        cfg.data.batch_size = args.batch_size
    if args.backbone:
        cfg.model.backbone = args.backbone

    seed_everything(cfg.seed)

    dm = SkinDiseaseDataModule(
        data_dir=cfg.data.data_dir,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.seed,
    )
    dm.setup()
    print(f"Discovered {dm.num_classes} classes: {dm.class_names}")

    model = build_model(
        backbone=cfg.model.backbone,
        num_classes=dm.num_classes,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
        freeze_backbone=cfg.model.freeze_backbone,
    )

    weights = dm.class_weights() if cfg.training.use_class_weights else None
    trainer = Trainer(model, cfg, dm.class_names, class_weights=weights)
    trainer.fit(dm.train_dataloader(), dm.val_dataloader())

    # Final evaluation on the test split using the best checkpoint.
    trainer.load_checkpoint(Path(cfg.output.checkpoint_dir) / "best.pth")
    y_true, y_pred = trainer.predict(dm.test_dataloader())
    metrics = classification_metrics(y_true, y_pred, dm.class_names)
    print(f"\nTest accuracy: {metrics['accuracy']:.4f} | macro-F1: {metrics['macro_f1']:.4f}")
    print(metrics["report_text"])

    plot_confusion_matrix(
        y_true,
        y_pred,
        dm.class_names,
        save_path=Path(cfg.output.log_dir) / "confusion_matrix.png",
    )
    print(f"Saved confusion matrix to {cfg.output.log_dir}/confusion_matrix.png")


if __name__ == "__main__":
    main()
