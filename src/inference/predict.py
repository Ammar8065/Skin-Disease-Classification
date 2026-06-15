"""Single-image / batch inference from a saved checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from ..data.transforms import build_transforms
from ..models.model import build_model
from ..utils.config import Config


class SkinDiseasePredictor:
    """Loads a checkpoint and classifies images."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.class_names: list[str] = ckpt["class_names"]
        cfg = Config.from_dict(ckpt["config"])

        self.model = build_model(
            backbone=cfg.model.backbone,
            num_classes=len(self.class_names),
            pretrained=False,
            dropout=cfg.model.dropout,
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        self.transform = build_transforms(cfg.data.image_size, train=False)

    @torch.no_grad()
    def predict(self, image_path: str | Path, top_k: int = 5) -> list[dict]:
        """Return the top-k (label, probability) predictions for one image."""
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(tensor), dim=1).squeeze(0)
        k = min(top_k, len(self.class_names))
        top_probs, top_idx = probs.topk(k)
        return [
            {"label": self.class_names[i], "probability": float(p)}
            for p, i in zip(top_probs.tolist(), top_idx.tolist())
        ]
