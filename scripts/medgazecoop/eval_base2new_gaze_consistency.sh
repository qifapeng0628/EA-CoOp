

DATA=$1
DATASET=$2
SHOTS=$3
MODEL=$4

export USE_GAZE_MASK=0

bash scripts/medgazecoop/eval_base2new.sh ${DATA} ${DATASET} ${SHOTS} ${MODEL}
