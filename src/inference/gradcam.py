"""Grad-CAM explainability for the skin disease classifier.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the image
regions most responsible for a prediction, by weighting the last conv feature
maps with the gradients of the target class score.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ..data.transforms import IMAGENET_MEAN, IMAGENET_STD
from ..models.model import build_model
from ..utils.config import Config


def _find_target_layer(model: nn.Module, backbone: str) -> nn.Module:
    """Return the last convolutional block to attach Grad-CAM hooks to."""
    b = backbone.lower()
    if b.startswith("resnet"):
        return model.layer4[-1]
    if b.startswith("efficientnet"):
        return model.features[-1]
    if b.startswith("mobilenet"):
        return model.features[-1]
    raise ValueError(
        f"Don't know the Grad-CAM target layer for backbone {backbone!r}. "
        "Add it to _find_target_layer()."
    )


class GradCAM:
    """Low-level Grad-CAM: hooks a target layer and produces a heatmap."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, module, inp, out) -> None:
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    def __call__(
        self, input_tensor: torch.Tensor, class_idx: int | None = None
    ) -> tuple[np.ndarray, int, torch.Tensor]:
        """Return (cam, class_idx, probabilities) for a single (1,C,H,W) input."""
        self.model.zero_grad()
        with torch.enable_grad():
            logits = self.model(input_tensor)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            logits[:, class_idx].sum().backward()

        # Weight each feature map by its mean gradient, sum, ReLU.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1,K,1,1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        probs = logits.detach().softmax(dim=1).squeeze(0).cpu()
        return cam, class_idx, probs

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


class SkinDiseaseGradCAM:
    """High-level wrapper: load a checkpoint and explain images with Grad-CAM."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.class_names: list[str] = ckpt["class_names"]
        cfg = Config.from_dict(ckpt["config"])
        self.image_size = cfg.data.image_size

        self.model = build_model(
            backbone=cfg.model.backbone,
            num_classes=len(self.class_names),
            pretrained=False,
            dropout=cfg.model.dropout,
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        target_layer = _find_target_layer(self.model, cfg.model.backbone)
        self.gradcam = GradCAM(self.model, target_layer)

        # Normalised tensor for the model.
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(self.image_size * 1.14)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        # Same spatial crop, but un-normalised, for display/overlay alignment.
        self.display_transform = transforms.Compose(
            [
                transforms.Resize(int(self.image_size * 1.14)),
                transforms.CenterCrop(self.image_size),
            ]
        )

    def explain(
        self,
        image_path: str | Path,
        class_idx: int | None = None,
        alpha: float = 0.5,
    ) -> dict:
        """Compute a Grad-CAM explanation for one image.

        Args:
            image_path: Path to the image.
            class_idx: Class to explain. Defaults to the model's top prediction.
            alpha: Heatmap blend strength for the overlay (0-1).

        Returns:
            dict with keys: label, probability, class_idx, image (HxWx3 float),
            cam (HxW float), overlay (HxWx3 float).
        """
        from matplotlib import cm

        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        cam, idx, probs = self.gradcam(input_tensor, class_idx)

        base = np.asarray(self.display_transform(image), dtype=np.float32) / 255.0
        heatmap = cm.jet(cam)[..., :3]
        overlay = np.clip((1.0 - alpha) * base + alpha * heatmap, 0.0, 1.0)

        return {
            "label": self.class_names[idx],
            "probability": float(probs[idx]),
            "class_idx": idx,
            "image": base,
            "cam": cam,
            "overlay": overlay,
        }
