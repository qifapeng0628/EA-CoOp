#!/bin/bash
# Evaluate few-shot model (no gaze at inference)
# Usage:
#   bash scripts/eval_fewshot_gaze_consistency.sh <DATA_ROOT> <DATASET> <SHOTS> <MODEL>

DATA=$1
DATASET=$2
SHOTS=$3
MODEL=$4

export USE_GAZE_MASK=0

bash scripts/biomedcoop/eval_fewshot.sh ${DATA} ${DATASET} ${SHOTS} ${MODEL}
