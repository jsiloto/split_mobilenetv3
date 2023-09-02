#!/bin/bash

PROJECT="supervised"
BASE_PARAMS="--hyper configs/hyper/supervised.yaml --project ${PROJECT} --clean --wandb"
BASE_PARAMS="${BASE_PARAMS} --dataset configs/dataset/pets.yaml"

for BETA in "0.6" "0.7" "0.8" "0.9"
do
    python train.py --model configs/model/entropy_bottleneck_1.0.yaml \
    $BASE_PARAMS --beta ${BETA}

done





