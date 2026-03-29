
set -e

DATA=$1
DATASET=$2
SHOTS=$3
MODEL=$4

export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# -----------------------
# Gaze: enable + heatmap root
# -----------------------
export USE_GAZE_MASK=1
export GAZE_HEATMAP_ROOT="${GAZE_HEATMAP_ROOT:-${DATA}/select/${DATASET}}"

# -----------------------
# Zoom-In Teacher (crop) settings (area-based)
# -----------------------
export GAZE_CROP_JITTER=${GAZE_CROP_JITTER:-0}

export GAZE_CROP_SMIN=${GAZE_CROP_SMIN:-32}
export GAZE_CROP_SMAX=${GAZE_CROP_SMAX:-224}
export GAZE_CROP_K=${GAZE_CROP_K:-1.0}
export GAZE_AREA_THR=${GAZE_AREA_THR:-0.35}
export GAZE_ADAPTIVE_CROP=${GAZE_ADAPTIVE_CROP:-1}

export GAZE_BINARIZE=${GAZE_BINARIZE:-0}
export GAZE_MASK_THR=${GAZE_MASK_THR:-0.03}

# -----------------------
# Loss weights
# -----------------------
export GAZE_CONS_LAMBDA=${GAZE_CONS_LAMBDA:-0.1}
export GAZE_CONS_T=${GAZE_CONS_T:-1.0}
export GAZE_CONS_WARMUP_EPOCHS=${GAZE_CONS_WARMUP_EPOCHS:-0}
export GAZE_ROI_CE_LAMBDA=${GAZE_ROI_CE_LAMBDA:-0.1}

# -----------------------
# SCGTA (optional) Not used !
# -----------------------
export GAZE_SCGTA_LAMBDA=${GAZE_SCGTA_LAMBDA:-0.0}
export GAZE_SCGTA_T=${GAZE_SCGTA_T:-1.0}

# -----------------------
# Sanity dump: save overlay & crop previews from dataloader
# -----------------------
export GAZE_SANITY_SAVE=${GAZE_SANITY_SAVE:-0}
export GAZE_SANITY_DIR=${GAZE_SANITY_DIR:-./gaze_sanity}
export GAZE_SANITY_MAX=${GAZE_SANITY_MAX:-16}

export GAZE_DEBUG=${GAZE_DEBUG:-0}
export GAZE_DEBUG_MAX=${GAZE_DEBUG_MAX:-20}

bash scripts/medgazecoop/few_shot.sh ${DATA} ${DATASET} ${SHOTS} ${MODEL}