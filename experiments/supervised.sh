#!/bin/bash

PROJECT="supervised"
BASE_PARAMS="--hyper configs/hyper/supervised.yaml --project ${PROJECT} --clean --wandb"

for BETA in "0.5" "1.0" "2.0" "4.0" "8.0" "16.0"
do
    python train.py --model configs/model/entropy_bottleneck_1.0.yaml \
    $BASE_PARAMS --beta ${BETA}
done





