#!/bin/bash

PROJECT="baseline"
BASE_PARAMS="--hyper configs/hyper/1epoch.yaml --project ${PROJECT} --clean"

python train.py --model configs/model/regular.yaml $BASE_PARAMS
python train.py --model configs/model/channel_bottleneck.yaml $BASE_PARAMS

PROJECT="baseline"

for BETA in "0.1" "1.0" "10.0"
do
  for RATIO in "0.1" "0.5" "1.0"
  do
    python train.py --model configs/model/entropy_bottleneck_${RATIO}_${BETA}.yaml $BASE_PARAMS
  done
done





