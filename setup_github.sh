#!/bin/bash
# =============================================================
# Script to update your GitHub repo with RetriVAD v2
# Run this on YOUR local machine after extracting RetriVAD_v2.tar.gz
# =============================================================

set -e

echo "Step 1: Clone your existing repo"
git clone https://github.com/hammadhaideer/RetriVAD.git RetriVAD_update
cd RetriVAD_update

echo "Step 2: Remove old files (keep .git)"
find . -maxdepth 1 ! -name '.git' ! -name '.' -exec rm -rf {} +

echo "Step 3: Copy new v2 files"
# Assuming RetriVAD_v2/ is in the parent directory
cp -r ../RetriVAD_v2/* .
cp ../RetriVAD_v2/.gitignore . 2>/dev/null || true

echo "Step 4: Stage and commit"
git add -A
git commit -m "RetriVAD v2: Fast Patch-Token Localisation + Coreset Memory Bank + Retrieval Explanation

Major update:
- Fast Patch-Token Localisation (FPL): pixel maps from single forward pass
- Coreset memory bank: greedy diversity-maximising selection
- Retrieval Explanation Module (REM): nearest-normal retrieval for explainability
- Cross-domain layer analysis across 9 datasets
- Clean modular codebase matching UniVAD (CVPR 2025) structure
- Support for all 9 benchmark datasets (industrial, logical, medical)"

echo "Step 5: Push"
git push origin main

echo "Done! Check https://github.com/hammadhaideer/RetriVAD"
