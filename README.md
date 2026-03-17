# RetriVAD: Training-Free Unified Visual Anomaly Detection via Single-Encoder Retrieval

---

## Introduction

RetriVAD is a training-free unified visual anomaly detection method that detects anomalies across industrial, logical, and medical imaging domains using a single frozen encoder, without any domain-specific training, fine-tuning, or GPU.

Given a category, RetriVAD builds a memory bank offline by encoding normal reference images into 768-dimensional feature vectors stored in a FAISS index. At test time, the anomaly score for a query image is its nearest-neighbour distance to the memory bank. No thresholds are tuned and no labels are used. The same index generalises across industrial products, brain MRI, chest X-rays, and retinal images.

Pixel-level localisation requires 256 sequential DINOv2 forward passes, one per 14×14-pixel crop from the 16×16 spatial grid, producing a 16×16 distance map which is bilinearly upsampled to the original image resolution.

---

## Results

### Image-Level AUROC (1-shot setting)

All methods evaluated on the same 7 datasets. UniVAD mean is recalculated on these 7 datasets from the per-dataset results reported in their paper (Gu et al., CVPR 2025).

| Dataset | PatchCore | WinCLIP | AnomalyGPT | UniAD | MedCLIP | UniVAD | RetriVAD |
|---------|-----------|---------|------------|-------|---------|--------|----------|
| MVTec-AD | 84.0 | 93.1 | 94.1 | 70.3 | 75.2 | 97.8 | 89.6 |
| VisA | 74.8 | 83.8 | 87.4 | 61.3 | 69.0 | 93.5 | 79.1 |
| MVTec LOCO | 62.0 | 58.0 | 60.4 | 50.9 | 54.9 | 71.0 | **72.1** |
| BrainMRI | 73.2 | 55.4 | 73.1 | 50.0 | 69.7 | 80.2 | **96.16** |
| RESC | 56.3 | 72.9 | 82.4 | 53.5 | 66.9 | 85.5 | 76.53 |
| ChestXray | 66.4 | 70.2 | 68.5 | 60.6 | 71.4 | 72.2 | **86.79** |
| OCT17 Kermany | 59.9 | 79.7 | 77.5 | 44.4 | 64.6 | 82.1 | 74.67 |
| **Mean (7 datasets)** | 68.1 | 73.3 | 77.6 | 55.9 | 67.4 | 83.2 | **82.1** |

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

Download from [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract to:

```
data/mvtec/
├── bottle/
├── cable/
└── ...
```

#### VisA

Download from [https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) and extract to:

```
data/VisA_20220922/
├── candle/
├── capsules/
└── ...
```

#### MVTec LOCO

Download from [https://www.mvtec.com/company/research/datasets/mvtec-loco](https://www.mvtec.com/company/research/datasets/mvtec-loco) and extract to:

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
| OCT17 Kermany | [Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/3) |

---

## Running Evaluation

### Image-Level Evaluation

```bash
python benchmark.py --dataset all --data_root /path/to/datasets
```

Runtime: approximately 30–60 minutes on CPU.

### Pixel-Level Evaluation

```bash
# MVTec-AD and MVTec LOCO
python scripts/eval_pixel_auroc.py --dataset mvtec --data_root /path/to/mvtec

# VisA
python eval_pixel_auroc_visa.py --data_root /path/to/VisA_20220922

# RESC
python eval_pixel_auroc_resc.py --data_root /path/to/resc/RESC
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

Output is saved to `figures/fig_qualitative.png`.

---

## Pre-computed Results

| File | Contents |
|------|----------|
| `results/image_auroc_results.json` | Image-level AUROC across all 7 datasets |
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
├── eval_oct17_kermany.py
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
