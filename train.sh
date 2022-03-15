#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config config/train.yaml \
#     >> outs_files/22_03_11-ntu120xsub-angular.log 2>&1 &