"""Class labels for the 22-class skin disease image classification dataset.

The canonical ordering below is fixed and must not be reordered: model
checkpoints, confusion matrices, and exported predictions all depend on the
integer index assigned to each class here.
"""

from __future__ import annotations

# Canonical, ordered list of class names. Index in this list == class id.
CLASS_NAMES: list[str] = [
    "Acne",
    "Actinic Keratosis",
    "Benign Tumors",
    "Bullous",
    "Candidiasis",
    "Drug Eruption",
    "Eczema",
    "Infestations/Bites",
    "Lichen",
    "Lupus",
    "Moles",
    "Psoriasis",
    "Rosacea",
    "Seborrheic Keratoses",
    "Skin Cancer",
    "Sun/Sunlight Damage",
    "Tinea",
    "Unknown/Normal",
    "Vascular Tumors",
    "Vasculitis",
    "Vitiligo",
    "Warts",
]

NUM_CLASSES: int = len(CLASS_NAMES)

# Bidirectional lookups.
LABEL_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL: dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}


def label_to_index(label: str) -> int:
    """Return the integer class id for a class name."""
    try:
        return LABEL_TO_INDEX[label]
    except KeyError as exc:
        raise KeyError(
            f"Unknown class label {label!r}. Expected one of: {CLASS_NAMES}"
        ) from exc


def index_to_label(index: int) -> str:
    """Return the class name for an integer class id."""
    try:
        return INDEX_TO_LABEL[index]
    except KeyError as exc:
        raise KeyError(
            f"Class index {index} out of range [0, {NUM_CLASSES - 1}]."
        ) from exc
