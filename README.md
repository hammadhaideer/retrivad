# RetriVAD: Training-Free Unified Visual Anomaly Detection via Single-Encoder Retrieval

---

## Introduction

RetriVAD is a training-free unified visual anomaly detection method that detects anomalies across industrial, logical, and medical imaging domains using a single frozen encoder, without any domain-specific training, fine-tuning, or GPU.

Given a category, RetriVAD builds a memory bank offline by encoding normal reference images into 768-dimensional feature vectors stored in a FAISS index. At test time, the anomaly score for a query image is its nearest-neighbour distance to the memory bank. No thresholds are tuned and no labels are used. The same index generalises across industrial products, brain MRI, chest X-rays, and retinal images.

Pixel-level localisation requires 256 sequential DINOv2 forward passes, one per 14x14-pixel crop from the 16x16 spatial grid, producing a 16x16 distance map which is bilinearly upsampled to the original image resolution.

---

## Results

### Image-Level AUROC (1-shot setting)

Evaluated on 6 datasets spanning industrial, logical, and medical domains.
All baseline numbers are taken from UniVAD (CVPR 2025) under the identical 1-normal-shot protocol.

| Dataset | PatchCore | AnomalyGPT | WinCLIP | ComAD | UniAD | MedCLIP | UniVAD | RetriVAD |
|---------|-----------|------------|---------|-------|-------|---------|--------|----------|
| MVTec-AD | 84.0 | 94.1 | 93.1 | 57.3 | 70.3 | 75.2 | **97.8** | 89.6 |
| VisA | 74.8 | 87.4 | 83.8 | 53.9 | 61.3 | 69.0 | **93.5** | 79.1 |
| MVTec LOCO | 62.0 | 60.4 | 58.0 | 62.2 | 50.9 | 54.9 | 71.0 | **72.1** |
| BrainMRI | 73.2 | 73.1 | 55.4 | 33.3 | 50.0 | 69.7 | 80.2 | **95.3** |
| RESC | 56.3 | 82.4 | 72.9 | 73.5 | 53.5 | 66.9 | **85.5** | 76.5 |
| ChestXray | 66.4 | 68.5 | 70.2 | 50.1 | 60.6 | 71.4 | 72.2 | **86.8** |
| **Mean** | 69.5 | 77.6 | 72.2 | 55.1 | 57.8 | 67.9 | 83.4 | **83.2** |

RetriVAD wins on 3 out of 6 datasets against UniVAD (CVPR 2025), matching its mean performance with a single frozen encoder and no GPU.
VisA results use the official CSV split protocol.

### Pixel-Level AUROC

| Dataset | Pixel-AUROC |
|---------|-------------|
| VisA (12 categories) | 91.92 |
| MVTec-AD (15 categories) | 71.27 |
| MVTec LOCO (5 categories) | 56.21 |

---

## Environment Setup

Clone the repository:

```bash
git clone https://github.com/hammadhaideer/RetriVAD.git
cd RetriVAD
```

Create and activate the conda environment:

```bash
conda create -n retrivad python=3.10
conda activate retrivad
```

Install PyTorch (CPU):

```bash
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies:

```
torch==2.10.0+cpu
faiss-cpu==1.13.2
scikit-learn==1.7.2
numpy==2.2.6
scipy
Pillow
tqdm
```

Verify installation:

```bash
python -c "import torch, faiss, numpy; print('torch:', torch.__version__); print('faiss: OK'); print('numpy:', numpy.__version__)"
```

---

## Data Preparation

Datasets are not included in this repository. Download each dataset and place it at the corresponding path.

#### MVTec-AD

Download from https://www.mvtec.com/company/research/datasets/mvtec-ad and extract to:

```
data/mvtec/
├── bottle/
├── cable/
└── ...
```

#### VisA

Download from https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar and extract to:

```
data/VisA_20220922/
├── candle/
├── capsules/
└── ...
```

#### MVTec LOCO

Download from https://www.mvtec.com/company/research/datasets/mvtec-loco and extract to:

```
data/mvtec_loco/
├── breakfast_box/
├── pushpins/
└── ...
```

#### Medical Datasets

| Dataset | Source |
|---------|--------|
| BrainMRI | [Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) |
| RESC | [GitHub](https://github.com/CharlesKing/RESC) |
| ChestXray | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |

---

## Running Evaluation

### Image-Level Evaluation

```bash
# All industrial and logical datasets
python benchmark.py --dataset mvtec --data_root /path/to/mvtec
python benchmark.py --dataset visa --data_root /path/to/VisA_20220922
python benchmark.py --dataset loco --data_root /path/to/mvtec_loco

# Medical datasets
python eval_brainmri.py --data_root /path/to/brain_mri
python eval_chestxray.py --data_root /path/to/chest_xray
python eval_resc.py --data_root /path/to/resc/RESC
```

### Pixel-Level Evaluation

```bash
# MVTec-AD and MVTec LOCO
python scripts/eval_pixel_auroc.py --dataset mvtec --data_root /path/to/mvtec

# VisA
python eval_pixel_auroc_visa.py --data_root /path/to/VisA_20220922
```

Pixel-level evaluation is computationally intensive on CPU-only hardware. All results are pre-computed and saved in `results/`. Re-running is only necessary if the model code changes.

### Latency Benchmark

```bash
python scripts/latency_benchmark.py --data_root /path/to/mvtec --category bottle --n_test 20
```

### Qualitative Figure

```bash
python generate_heatmaps.py \
    --carpet_normal  /path/to/mvtec/carpet/train/good/000.png \
    --carpet_test    /path/to/mvtec/carpet/test/color/000.png \
    --carpet_gt      /path/to/mvtec/carpet/ground_truth/color/000_mask.png \
    --bottle_normal  /path/to/mvtec/bottle/train/good/000.png \
    --bottle_test    /path/to/mvtec/bottle/test/broken_large/000.png \
    --bottle_gt      /path/to/mvtec/bottle/ground_truth/broken_large/000_mask.png \
    --brain_normal   /path/to/brain_mri/Training/notumor/Tr-no_999.jpg \
    --brain_test     /path/to/brain_mri/Testing/glioma/Te-gl_1.jpg \
    --visa_normal    /path/to/VisA_20220922/pcb1/Data/Images/Normal/0000.JPG \
    --visa_test      /path/to/VisA_20220922/pcb1/Data/Images/Anomaly/009.JPG \
    --visa_gt        /path/to/VisA_20220922/pcb1/Data/Masks/Anomaly/009.png \
    --output         figures/fig_qualitative.png
```

---

## Pre-computed Results

| File | Contents |
|------|----------|
| `results/image_auroc_results.json` | Image-level AUROC across all 6 datasets |
| `results/mvtec_ad_retrivad.json` | Per-category results for MVTec-AD |
| `results/visa_retrivad.json` | Per-category results for VisA |
| `results/mvtecloco_retrivad.json` | Per-category results for MVTec LOCO |
| `results/pixel_auroc_final.json` | Pixel-level AUROC for MVTec-AD and MVTec LOCO |
| `results/visa_pixel_auroc.json` | Pixel-level AUROC for VisA (12 categories) |
| `results/latency_results.json` | Per-image latency measured on CPU |

---

## Repository Structure

```
RetriVAD/
├── README.md
├── requirements.txt
├── benchmark.py
├── generate_heatmaps.py
├── ablation_features.py
├── eval_brainmri.py
├── eval_chestxray.py
├── eval_resc.py
├── eval_pixel_auroc_visa.py
├── eval_pixel_auroc_resc.py
├── models/
│   └── retrivad.py
├── utils/
│   └── metrics.py
├── scripts/
│   ├── eval_pixel_auroc.py
│   └── latency_benchmark.py
├── results/
└── figures/
```

---

## Citation

```bibtex
@article{haider2026retrivad,
  title={RetriVAD: Training-Free Unified Visual Anomaly Detection via Single-Encoder Retrieval},
  author={Haider, Hammad Ali and Zheng, Panpan},
  year={2026}
}
```
