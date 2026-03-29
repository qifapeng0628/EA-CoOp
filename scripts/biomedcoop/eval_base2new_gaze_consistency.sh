#!/bin/bash
# Evaluate base-to-new model (no gaze at inference)
# Usage:
#   bash scripts/eval_base2new_gaze_consistency.sh <DATA_ROOT> <DATASET> <SHOTS> <MODEL>

DATA=$1
DATASET=$2
SHOTS=$3
MODEL=$4

export USE_GAZE_MASK=0

bash scripts/biomedcoop/eval_base2new.sh ${DATA} ${DATASET} ${SHOTS} ${MODEL}
