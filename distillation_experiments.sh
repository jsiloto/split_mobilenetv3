#!/bin/bash

PROJECT="distillation"
BASE_PARAMS="--hyper configs/hyper/default.yaml --project ${PROJECT} --clean --wandb"

for BETA in "5.0" "50.0"
do
    python train.py --student configs/model/entropy_bottleneck_1.0_${BETA}.yaml \
    --teacher configs/model/channel_bottleneck_1.0.yaml $BASE_PARAMS
done





