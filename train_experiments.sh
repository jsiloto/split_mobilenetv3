#!/bin/bash

PROJECT="train1"
BASE_PARAMS="--hyper configs/hyper/default.yaml --project ${PROJECT} --clean --wandb"

for BETA in "1.0" "5.0" "25.0" "10.0" "50.0"
do
    python train.py --model configs/model/entropy_bottleneck_1.0_${BETA}.yaml $BASE_PARAMS
done





