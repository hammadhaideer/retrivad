#!/bin/bash
# RetriVAD Evaluation Script
# Usage: bash test.sh [dataset] [data_path]
#
# Examples:
#   bash test.sh mvtec ./data/mvtec
#   bash test.sh all ./data

DATASET=${1:-mvtec}
DATA_PATH=${2:-./data}
DEVICE=${3:-cpu}
K_SHOT=${4:-69}

echo "============================================"
echo "RetriVAD Evaluation"
echo "  Dataset:  ${DATASET}"
echo "  Data:     ${DATA_PATH}"
echo "  Device:   ${DEVICE}"
echo "  K-shot:   ${K_SHOT}"
echo "============================================"

# Main evaluation
python test_retrivad.py \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --device ${DEVICE} \
    --k_shot ${K_SHOT} \
    --use_coreset \
    --save_dir ./results

echo ""
echo "Done. Results saved to ./results/"
