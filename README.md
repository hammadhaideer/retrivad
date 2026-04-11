# RetriVAD: Training-Free Unified Visual Anomaly Detection via Single-Encoder Retrieval

Official implementation of the paper "RetriVAD: Training-Free Unified Visual Anomaly Detection via Single-Encoder Retrieval with Fast Patch-Token Localisation."

## Introduction

RetriVAD is a training-free visual anomaly detection method that uses a single frozen DINOv2 ViT-B/14 encoder and FAISS nearest-neighbour retrieval to detect anomalies across industrial, logical, and medical domains without any domain-specific adaptation. Given a few normal reference images, RetriVAD builds a dual memory bank (image-level and patch-level) and scores query images by their distance to the nearest normal reference. It consists of three components:

- **Fast Patch-Token Localisation (FPL)**: Produces a 16×16 pixel-level anomaly map from the 256 patch tokens of a single DINOv2 forward pass, eliminating the need for additional encoder calls.
- **Coreset Memory Bank (CMB)**: Selects reference images via greedy maximum-coverage sampling to improve few-shot generalisation.
- **Retrieval Explanation Module (REM)**: Retrieves the most similar normal references for each query and highlights patch-level differences as visual anomaly evidence.

Experiments on nine datasets spanning three domains show that RetriVAD matches the performance of multi-model pipelines while operating entirely on CPU at 0.76 seconds per image.

## Overview of RetriVAD

<p align="center">
  <img src="figures/architecture.png" width="800">
</p>

## Running RetriVAD

### Environment Installation

Clone the repository:
```
git clone https://github.com/hammadhaideer/RetriVAD.git
cd RetriVAD
```

Install the required packages:
```
pip install -r requirements.txt
```

### Prepare Data

#### MVTec-AD
- Download and extract [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) into `data/mvtec`
- Run `python data/generate_meta.py --dataset mvtec --data_root data/mvtec` to obtain `data/mvtec/meta.json`

#### VisA
- Download and extract [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)
- Refer to the instructions in [spot-diff](https://github.com/amazon-science/spot-diff?tab=readme-ov-file#data-preparation) to get the 1-class format and put it into `data/VisA_pytorch/1cls`

#### MVTec LOCO AD
- Download the [MVTec LOCO Caption](https://github.com/hujiecpp/MVTec-Caption) dataset and put it into `data/mvtec_loco_caption`

#### Medical Datasets
- The medical datasets are obtained from [BMAD](https://github.com/DorisBao/BMAD) and organised in MVTec format
- Download from [this OneDrive link](https://1drv.ms/u/s!AopsN_HMhJeckoJT-3yF_pwQMSn9OA?e=nRW1wA) and put them into `data/`

#### Data Format
The prepared data format should be as follows:
```
data
├── mvtec
│   ├── meta.json
│   ├── bottle
│   ├── cable
│   ├── ...
├── VisA_pytorch/1cls
│   ├── meta.json
│   ├── candle
│   ├── capsules
│   ├── ...
├── mvtec_loco_caption
│   ├── meta.json
│   ├── breakfast_box
│   ├── juice_bottle
│   ├── ...
├── BrainMRI
│   ├── meta.json
│   ├── train
│   ├── test
│   ├── ground_truth
├── LiverCT
│   ├── meta.json
│   ├── train
│   ├── test
│   ├── ground_truth
├── RESC
│   ├── meta.json
│   ├── train
│   ├── test
│   ├── ground_truth
├── HIS
│   ├── meta.json
│   ├── train
│   ├── test
├── ChestXray
│   ├── meta.json
│   ├── train
│   ├── test
├── OCT17
│   ├── meta.json
│   ├── train
│   ├── test
```

### Run the Test Script

Evaluate on a single dataset:
```
python test_retrivad.py --dataset mvtec --data_path ./data/mvtec --k_shot 69
```

Evaluate on all nine datasets:
```
python test_retrivad.py --dataset all --data_path ./data --k_shot 69
```

Skip pixel-level evaluation for faster results:
```
python test_retrivad.py --dataset mvtec --data_path ./data/mvtec --no_pixel
```

### Ablation Experiments

Shot count ablation (coreset vs random selection):
```
python run_ablation.py --experiment shots --dataset mvtec --data_path ./data/mvtec
```

Feature configuration ablation:
```
python run_ablation.py --experiment features --dataset mvtec --data_path ./data/mvtec
```

Cross-domain layer analysis:
```
python run_layer_analysis.py --dataset mvtec --data_path ./data/mvtec
```

Multi-seed evaluation (5 seeds, mean ± std):
```
python run_statistics.py --dataset mvtec --data_path ./data/mvtec
```

## Citation

If you find RetriVAD useful in your research, please cite:
```
@inproceedings{haider2026retrivad,
  title={RetriVAD: Training-Free Unified Visual Anomaly Detection via Single-Encoder Retrieval with Fast Patch-Token Localisation},
  author={Haider, Hammad Ali and Zheng, Panpan},
  booktitle={Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM)},
  year={2026}
}
```

## Acknowledgements

This work builds upon [UniVAD](https://github.com/FantasticGNU/UniVAD) (CVPR 2025), [DINOv2](https://github.com/facebookresearch/dinov2), and [FAISS](https://github.com/facebookresearch/faiss).
