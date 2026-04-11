# RetriVAD: Training-Free Unified Visual Anomaly Detection via Single-Encoder Retrieval with Fast Patch-Token Localisation

Official implementation of **RetriVAD** (CIKM 2026).

## Introduction

RetriVAD is a training-free visual anomaly detection framework that replaces complex multi-model inference pipelines with a **single frozen DINOv2 ViT-B/14 encoder** and **FAISS nearest-neighbour retrieval**. It achieves competitive performance with the CVPR 2025 state-of-the-art (UniVAD) while running entirely on CPU.

**Key features:**
- **Training-free**: No fine-tuning, no gradient computation at any stage
- **Single encoder**: Frozen DINOv2 ViT-B/14 (86.6M parameters)
- **CPU-only inference**: 0.76s per image on standard dual-core CPU
- **Fast Patch-Token Localisation (FPL)**: Pixel-level anomaly maps from a single forward pass (zero extra encoder calls)
- **Coreset memory bank**: Diversity-maximising reference selection
- **Retrieval explanation**: Human-auditable anomaly evidence via nearest normal retrieval

## Overview

<p align="center">
  <img src="figures/architecture.png" width="800">
</p>

## Results

### Image-Level AUROC (%) — 1-Normal-Shot Setting

| Dataset | PatchCore | AnomalyGPT | WinCLIP | UniVAD | **RetriVAD** |
|---------|-----------|------------|---------|--------|-------------|
| MVTec-AD | 84.0 | 94.1 | 93.1 | **97.8** | — |
| VisA | 74.8 | 87.4 | 83.8 | **93.5** | — |
| MVTec LOCO | 62.0 | 60.4 | 58.0 | 71.0 | — |
| BrainMRI | 73.2 | 73.1 | 55.4 | 80.2 | — |
| RESC | 56.3 | 82.4 | 72.9 | **85.5** | — |
| ChestXray | 66.4 | 68.5 | 70.2 | 72.2 | — |

*Results will be updated after full evaluation.*

## Running RetriVAD

### Environment Installation

```bash
git clone https://github.com/hammadhaideer/RetriVAD.git
cd RetriVAD
pip install -r requirements.txt
```

### Prepare Data

#### MVTec-AD
- Download [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) into `data/mvtec`
- Run `python data/generate_meta.py --dataset mvtec --data_root data/mvtec`

#### VisA
- Download [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)
- Follow [instructions](https://github.com/amazon-science/spot-diff?tab=readme-ov-file#data-preparation) for 1-class format
- Place in `data/VisA_pytorch/1cls`

#### MVTec LOCO
- Download [MVTec LOCO Caption](https://github.com/hujiecpp/MVTec-Caption) into `data/mvtec_loco_caption`

#### Medical Datasets
- Download from [BMAD benchmark](https://github.com/DorisBao/BMAD) or [OneDrive](https://1drv.ms/u/s!AopsN_HMhJeckoJT-3yF_pwQMSn9OA?e=nRW1)
- Supported: BrainMRI, LiverCT, RESC, HIS, ChestXray, OCT17

### Evaluation

```bash
# Single dataset
python test_retrivad.py --dataset mvtec --data_path ./data/mvtec --k_shot 69

# All datasets
python test_retrivad.py --dataset all --data_path ./data --k_shot 69

# Without pixel-level evaluation (faster)
python test_retrivad.py --dataset mvtec --data_path ./data/mvtec --no_pixel

# Using shell script
bash test.sh mvtec ./data/mvtec
```

### Ablation Experiments

```bash
# Shot count ablation (coreset vs random)
python run_ablation.py --experiment shots --dataset mvtec --data_path ./data/mvtec

# Feature configuration ablation
python run_ablation.py --experiment features --dataset mvtec --data_path ./data/mvtec

# Layer analysis
python run_layer_analysis.py --dataset mvtec --data_path ./data/mvtec

# Multi-seed evaluation (5 seeds)
python run_statistics.py --dataset mvtec --data_path ./data/mvtec
```

## Project Structure

```
RetriVAD/
├── RetriVAD.py                 # Main RetriVAD class
├── test_retrivad.py            # Evaluation script
├── run_ablation.py             # Ablation experiments
├── run_layer_analysis.py       # Cross-domain layer analysis
├── run_statistics.py           # Multi-seed evaluation
├── test.sh                     # Shell runner
├── requirements.txt
├── models/
│   ├── encoder.py              # DINOv2 wrapper (CLS + patch tokens)
│   ├── memory_bank.py          # Dual FAISS index + coreset selection
│   ├── scoring.py              # Image scoring + Fast Patch Localisation
│   └── explanation.py          # Retrieval Explanation Module
├── datasets/
│   └── base.py                 # Unified loader for all 9 datasets
├── data/
│   └── generate_meta.py        # Generate UniVAD-compatible meta.json
├── utils/
│   └── metrics.py              # AUROC computation
└── results/                    # Output directory
```

## Citation

```bibtex
@inproceedings{haider2026retrivad,
  title={RetriVAD: Training-Free Unified Visual Anomaly Detection via Single-Encoder Retrieval with Fast Patch-Token Localisation},
  author={Haider, Hammad Ali and Zheng, Panpan},
  booktitle={CIKM},
  year={2026}
}
```

## Acknowledgements

This work builds upon [UniVAD](https://github.com/FantasticGNU/UniVAD) (CVPR 2025), [DINOv2](https://github.com/facebookresearch/dinov2), and [FAISS](https://github.com/facebookresearch/faiss).
