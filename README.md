# Skin Disease Image Classifier

Transfer-learning image classifier for a **22-class skin disease dataset**
(Acne, Eczema, Psoriasis, Melanoma/Skin Cancer, Vitiligo, …). Designed to be
**trained on Google Colab** — the dataset is pulled from Kaggle directly onto
the Colab VM, so nothing needs to be downloaded to your local machine.

> ⚠️ **Medical disclaimer:** This is a portfolio/research project. It is **not**
> a diagnostic tool and must not be used for clinical decisions.

---

## Project structure

```
.
├── configs/
│   └── default.yaml          # All hyperparameters / paths
├── notebooks/
│   └── train_colab.ipynb     # End-to-end Colab training notebook
├── scripts/
│   └── train.py              # CLI training entry point
├── src/
│   ├── data/
│   │   ├── labels.py         # Canonical 22 class names + lookups
│   │   ├── transforms.py     # Train/eval image transforms
│   │   └── datamodule.py     # Layout-agnostic dataset + dataloaders
│   ├── models/
│   │   └── model.py          # torchvision backbones + custom head
│   ├── training/
│   │   ├── trainer.py        # AMP, scheduling, class weights, early stop
│   │   └── metrics.py        # Accuracy / F1 / confusion matrix
│   ├── inference/
│   │   └── predict.py        # Load checkpoint -> predict single image
│   └── utils/
│       ├── config.py         # Typed YAML config (dataclasses)
│       └── seed.py           # Reproducibility
├── requirements.txt
└── .gitignore
```

## Quick start (Google Colab) — recommended

1. Push this project to a GitHub repo.
2. Open `notebooks/train_colab.ipynb` in Colab and select a **GPU** runtime.
3. Set `REPO_URL` to your repo, run the cells top to bottom.
4. Provide Kaggle credentials — either add `KAGGLE_USERNAME` / `KAGGLE_KEY` as
   Colab secrets (recommended), or upload your `kaggle.json` when prompted
   (Kaggle → Account → *Create New API Token*).
5. The dataset (`pacificrm/skindiseasedataset`) is fetched via `kagglehub`,
   then the notebook trains, evaluates, plots a confusion matrix, and saves the
   best checkpoint to your Google Drive.

## Quick start (local)

```bash
pip install -r requirements.txt

# Download the dataset via kagglehub (uses ~/.kaggle/kaggle.json or env vars):
python -c "import kagglehub; print(kagglehub.dataset_download('pacificrm/skindiseasedataset'))"

# Then point training at the printed path:
python -m scripts.train --config configs/default.yaml --data-dir /path/from/above
```

The data module auto-detects the dataset layout:
- `train/` + `test/` (and optional `val/`) subfolders, **or**
- a single folder with one subfolder per class (an automatic stratified
  train/val/test split is created).

## Inference

```python
from src.inference import SkinDiseasePredictor

predictor = SkinDiseasePredictor("outputs/checkpoints/best.pth")
print(predictor.predict("some_image.jpg", top_k=5))
```

## Configuration

Everything is driven by [configs/default.yaml](configs/default.yaml) — backbone,
image size, batch size, optimizer/scheduler, class weighting, early stopping,
mixed precision, and output paths. Override fields on the CLI (`--epochs`,
`--batch-size`, `--backbone`) or directly in the notebook.

## The 22 classes

Acne · Actinic Keratosis · Benign Tumors · Bullous · Candidiasis · Drug
Eruption · Eczema · Infestations/Bites · Lichen · Lupus · Moles · Psoriasis ·
Rosacea · Seborrheic Keratoses · Skin Cancer · Sun/Sunlight Damage · Tinea ·
Unknown/Normal · Vascular Tumors · Vasculitis · Vitiligo · Warts
