"""Image transforms for training and evaluation.

Uses ImageNet normalisation statistics because the backbones are pretrained
on ImageNet.
"""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    """Build a transform pipeline.

    Training pipeline includes light augmentation suited to dermatology images
    (flips, small rotations, colour jitter). Evaluation pipeline is a
    deterministic resize + center crop.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    resize = int(image_size * 1.14)  # resize a bit larger, then center crop
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
