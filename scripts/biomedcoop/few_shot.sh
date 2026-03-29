#!/bin/bash

# custom config
DATA=$1 #取命令行第一个参数，这里是data的路径名称
DATASET=$2 #取命令行第二个参数，这里是数据集的名称
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
MODEL=$4 #取命令行第四个参数，使用的LLM名称（BiomedCLIP）
NCTX=4  #上下文token的数量（prompt中token的数量）
CSC=False # 是否使用 class-specific context（类别特定上下文）
CTP=end #class token 在 prompt 中的位置（例如放在末尾）

METHOD=BiomedCoOp #表示所使用的算法
TRAINER=BiomedCoOp_${MODEL} # 是完整的训练器类名，依赖于模型类型

for SEED in 1 2 3
do
        DIR=output/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
           python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml  \
            --output-dir ${DIR} \
            TRAINER.BIOMEDCOOP.N_CTX ${NCTX} \
            TRAINER.BIOMEDCOOP.CSC ${CSC} \
            TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
        fi
done