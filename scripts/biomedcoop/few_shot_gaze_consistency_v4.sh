# #!/bin/bash
# # few_shot + gaze consistency (train-time only)
# # Usage:
# #   bash scripts/few_shot_gaze_consistency_v4.sh <DATA_ROOT> <DATASET> <SHOTS> <MODEL>
# #
# # Expected heatmap layout:
# #   ${DATA_ROOT}/select/${DATASET}/<classname>/<stem>_heatmap.png

# DATA=$1
# DATASET=$2
# SHOTS=$3
# MODEL=$4


# export TRANSFORMERS_OFFLINE=1
# export CUDA_VISIBLE_DEVICES=0
# # -----------------------
# # Gaze: enable + heatmap root
# # -----------------------
# export USE_GAZE_MASK=1
# export GAZE_HEATMAP_ROOT="${DATA}/select/${DATASET}"



# # -----------------------
# # Zoom-In Teacher (crop) settings
# # -----------------------
# export GAZE_CROP_SIZE=${GAZE_CROP_SIZE:-192}          # crop window on the 224x224 transformed image
# export GAZE_CROP_JITTER=${GAZE_CROP_JITTER:-0}        # optional center jitter in pixels (train only)
# export GAZE_BG_QUANTILE=${GAZE_BG_QUANTILE:-0.2}      # for bg-crop sampling (0.2 => pick from lowest 20% heatmap area)

# export GAZE_CROP_SMIN=${GAZE_CROP_SMIN:-32}      # 最小框，默认 32；BTMRI 病灶更小可设 16~32
# export GAZE_CROP_SMAX=${GAZE_CROP_SMAX:-224}      # 不设则用 GAZE_CROP_SIZE
# export GAZE_CROP_K=${GAZE_CROP_K:-1.5}        # 上下文比例系数 k，默认 1.5（你公式里的 k）
# export GAZE_AREA_THR=${GAZE_AREA_THR:-0.5}      # “红区”阈值，默认 0.6（越大越只看最红核心）
# export GAZE_ADAPTIVE_CROP=${GAZE_ADAPTIVE_CROP:-1}   # 1=启用自适应；0=退化为固定 crop

# # Keep soft heatmap by default (recommended for stable center-of-mass)
# export GAZE_BINARIZE=${GAZE_BINARIZE:-0}
# export GAZE_MASK_THR=${GAZE_MASK_THR:-0.03}           # used only if GAZE_BINARIZE=1

# # -----------------------
# # Loss weights (your template)
# # -----------------------

# export GAZE_CONS_LAMBDA=${GAZE_CONS_LAMBDA:-0.1}
# export GAZE_CONS_T=${GAZE_CONS_T:-1.0}
# export GAZE_CONS_WARMUP_EPOCHS=${GAZE_CONS_WARMUP_EPOCHS:-0}

# # -----------------------
# # Sanity dump: save overlay & crop previews from dataloader
# # -----------------------
# export GAZE_SANITY_SAVE=${GAZE_SANITY_SAVE:-0}        # set 1 to enable
# export GAZE_SANITY_DIR=${GAZE_SANITY_DIR:-./gaze_sanity}
# export GAZE_SANITY_MAX=${GAZE_SANITY_MAX:-16}

# # Optional: verbose prints for mask stats
# export GAZE_DEBUG=${GAZE_DEBUG:-0}
# export GAZE_DEBUG_MAX=${GAZE_DEBUG_MAX:-20}

# export GAZE_ROI_CE_LAMBDA=${GAZE_ROI_CE_LAMBDA:-0.1}

# bash scripts/biomedcoop/few_shot.sh ${DATA} ${DATASET} ${SHOTS} ${MODEL}



#!/usr/bin/env bash
# few_shot + gaze consistency (train-time only) — MedgazeCoOp v5
# Usage:
#   bash scripts/few_shot_gaze_consistency_v4.sh <DATA_ROOT> <DATASET> <SHOTS> <MODEL>
#
# Expected heatmap layout:
#   ${DATA_ROOT}/select/${DATASET}/<classname>/<stem>_heatmap.png
#
# Changes from v4:
#   - Crop sizing uses second-order moments (σx, σy) instead of threshold area
#   - RandAugment applied with paired spatial ops on mask
#   - Crop pre-computed in DatasetWrapper (not in model forward)
#   - SCCM/KDSP removed from model
#   - Renamed to MedgazeCoOp_BiomedCLIP

set -e

DATA=$1
DATASET=$2
SHOTS=$3
MODEL=$4

export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# -----------------------
# Gaze: enable + heatmap root
# -----------------------
export USE_GAZE_MASK=1
export GAZE_HEATMAP_ROOT="${DATA}/select/${DATASET}"

# -----------------------
# Second-order moment adaptive crop settings
# Crop size = clip(k * 2 * sqrt(σx² + σy²), smin, smax)
# where σx, σy are std devs of the heatmap distribution
# -----------------------
export GAZE_CROP_K=${GAZE_CROP_K:-1.5}            # context multiplier k
export GAZE_CROP_SMIN=${GAZE_CROP_SMIN:-32}        # minimum crop size
export GAZE_CROP_SMAX=${GAZE_CROP_SMAX:-224}       # maximum crop size
export GAZE_CROP_JITTER=${GAZE_CROP_JITTER:-0}     # optional center jitter in pixels (train only)

# Keep soft heatmap by default (recommended for stable center-of-mass & moments)
export GAZE_BINARIZE=${GAZE_BINARIZE:-0}
export GAZE_MASK_THR=${GAZE_MASK_THR:-0.03}        # used only if GAZE_BINARIZE=1

# -----------------------
# Loss weights
# -----------------------
export GAZE_CONS_LAMBDA=${GAZE_CONS_LAMBDA:-0.1}
export GAZE_CONS_T=${GAZE_CONS_T:-1.0}
export GAZE_CONS_WARMUP_EPOCHS=${GAZE_CONS_WARMUP_EPOCHS:-0}

export GAZE_ROI_CE_LAMBDA=${GAZE_ROI_CE_LAMBDA:-0.1}

# -----------------------
# Sanity dump: save overlay & crop previews from dataloader
# -----------------------
export GAZE_SANITY_SAVE=${GAZE_SANITY_SAVE:-0}     # set 1 to enable
export GAZE_SANITY_DIR=${GAZE_SANITY_DIR:-./gaze_sanity}
export GAZE_SANITY_MAX=${GAZE_SANITY_MAX:-16}

# Optional: verbose prints for mask stats
export GAZE_DEBUG=${GAZE_DEBUG:-0}
export GAZE_DEBUG_MAX=${GAZE_DEBUG_MAX:-20}

bash scripts/biomedcoop/few_shot.sh ${DATA} ${DATASET} ${SHOTS} ${MODEL}