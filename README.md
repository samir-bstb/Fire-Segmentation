# 🔥 Wildfire Semantic Segmentation — Comparative Study

A systematic comparison of five deep learning architectures for pixel-level wildfire segmentation, evaluated across three benchmark datasets under varying data availability conditions (25%, 50%, and 100% of training data).

---

## 📌 Overview

This repository contains the full implementation, training notebooks, and evaluation results for a comparative study between:

- **UNet** (ResNet34 encoder + InceptionV3 encoder)
- **SegFormer-B2** (Mix Transformer encoder, pretrained on ADE20K)
- **DeepLabV3+** (ResNet50 encoder)
- **EViT-UNet** (Efficient Vision Transformer UNet)
- **STUNet + DenseFire**

All models are trained and evaluated using 25%, 50%, and 100% of each dataset to assess segmentation performance under limited data conditions.

---

## 📂 Datasets

| Dataset | Source | Description |
|---|---|---|
| **Corsican Fire Dataset** | University of Corsica | Ground-level fire images with pixel-level annotations |
| **Boreal Forest Fire** | UAV-collected | Wildfire detection and smoke segmentation from drone imagery |
| **Fire Segmentation Image Dataset** | DI-VERSISai via Kaggle | RGB fire images with binary segmentation masks |

---

## 🧠 Models

### UNet (ResNet34 / InceptionV3)
Classical encoder-decoder architecture with skip connections. Two encoder variants evaluated: ResNet34 (4 symmetric downsampling stages, naturally suited for UNet) and InceptionV3 (multi-scale parallel convolutions, adapted as encoder).

### SegFormer-B2
Transformer-based segmentation model using a hierarchical Mix Transformer (MiT-B2) encoder and a lightweight MLP decoder. Pretrained on ADE20K segmentation data, providing a strong initialization advantage over ImageNet-pretrained CNN encoders.

### DeepLabV3+
Atrous (dilated) convolution architecture with an Atrous Spatial Pyramid Pooling (ASPP) module for multi-scale context aggregation, combined with a decoder that refines segmentation boundaries using low-level features.

### EViT-UNet
UNet variant with an Efficient Vision Transformer encoder, combining the spatial inductive bias of UNet skip connections with the global attention of transformer encoders.

### STUNet + DenseFire
Swin Transformer UNet augmented with DenseFire feature aggregation modules, designed to leverage dense connectivity patterns for improved fire region delineation.

---

## 📊 Evaluation Metrics

All models are evaluated on the held-out test split using:

- **F1 Score** — harmonic mean of precision and recall
- **mIoU** — mean Intersection over Union
- **MCC** — Matthews Correlation Coefficient
- **HAF** — Hamming Accuracy Factor
- **Dice Loss**

---

## 🗂️ Repository Structure

```
├── notebooks/
│   ├── UNet_ResNet34_kaggle.ipynb
│   ├── UNet_ResNet34_corsican.ipynb
│   ├── UNet_ResNet34_boreal.ipynb
│   ├── UNet_InceptionV3_kaggle.ipynb
│   ├── UNet_InceptionV3_corsican.ipynb
│   ├── SegFormer_kaggle.ipynb
│   ├── SegFormer_corsican.ipynb
│   ├── SegFormer_boreal.ipynb
│   ├── DeepLabV3Plus_kaggle.ipynb
│   ├── EViT_UNet.ipynb
│   └── STUNet_DenseFire.ipynb
├── results/
│   ├── *.json               # Test results per model and split
│   └── *.pth                # Saved model checkpoints
└── README.md
```

---

## ⚙️ Training Setup

| Parameter | Value |
|---|---|
| Image size | 256 × 256 |
| Batch size | 8 |
| Epochs | 50 (with early stopping, patience=10) |
| Optimizer | Adam (lr=1e-4) |
| Loss function | Dice Loss |
| Train / Val / Test split | 70% / 15% / 15% |
| Random seed | 42 |

Data augmentation applied during training: horizontal flip, vertical flip, rotation, brightness/contrast adjustment, and normalization (ImageNet mean/std).

---

## 🚀 Usage

All notebooks are designed to run on **Google Colab** with GPU acceleration. Models and results are automatically saved to Google Drive to avoid re-training on subsequent runs.

```python
# Each notebook follows the same structure:
train_25, val_25, test_25 = create_data_splits(..., data_percentage=0.25)
history_25, test_results_25, model_25 = train_model(..., data_percentage=0.25)

train_50, val_50, test_50 = create_data_splits(..., data_percentage=0.50)
history_50, test_results_50, model_50 = train_model(..., data_percentage=0.50)

train_100, val_100, test_100 = create_data_splits(..., data_percentage=1.0)
history_100, test_results_100, model_100 = train_model(..., data_percentage=1.0)
```

If a trained model already exists in the save directory, it is loaded automatically without re-training.

---

## 📦 Dependencies

```
torch
torchvision
segmentation-models-pytorch
transformers
albumentations
numpy
Pillow
matplotlib
```

Install via:
```bash
pip install segmentation-models-pytorch transformers albumentations
```

---

## 📄 Citation

If you use this code or results in your work, please cite accordingly. Dataset citations:

- **Corsican Fire Dataset**: Toulouse, G. et al. (2017). Computer vision for wildfire research: An evolving image dataset for benchmarking. *Fire Safety Journal*.
- **Boreal Forest Fire Dataset**: UAV-collected wildfire detection and smoke segmentation dataset.
- **Fire Segmentation Image Dataset**: DI-VERSISai. Available at: https://www.kaggle.com

---

## 📬 Contact

For questions or collaboration, open an issue in this repository.
