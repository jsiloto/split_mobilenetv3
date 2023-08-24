#!/bin/bash

PROJECT="supervised_a2"
BASE_PARAMS="--hyper configs/hyper/supervised.yaml --project ${PROJECT} --clean --wandb"
BASE_PARAMS="${BASE_PARAMS} --dataset configs/dataset/pets_a2.yaml"

for BETA in "0.1" "0.2" "0.3" "0.4" "0.5" "0.75" "1.0"
do
    python train.py --model configs/model/entropy_bottleneck_1.0.yaml \
    $BASE_PARAMS --beta ${BETA}

done

PROJECT="supervised_b2"
BASE_PARAMS="--hyper configs/hyper/supervised.yaml --project ${PROJECT} --clean --wandb"
BASE_PARAMS="${BASE_PARAMS} --dataset configs/dataset/pets_b2.yaml"

for BETA in "0.1" "0.2" "0.3" "0.4" "0.5" "0.75" "1.0"
do
    python train.py --model configs/model/entropy_bottleneck_1.0.yaml \
    $BASE_PARAMS --beta ${BETA}

done