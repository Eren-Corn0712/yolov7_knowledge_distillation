#!/bin/bash

function yolo() {
  python3 distill_train.py \
    --device 2 \
    --workers 16 \
    --batch-size 64 \
    --epochs 600 \
    --data data/bdd100k.yaml \
    --img 640 640 \
    --cfg cfg/yolov7_10c/yolov7-w01-10c.yaml \
    --hyp data/hyp.scratch.v7.w0.1.10c.kd.sckd.yaml \
    --teacher-weights runs/yolov7/yolov7_w_10_10c_e600/weights/best.pt \
    --weights 'yolov7.pt' \
    --project runs/yolov7_kd \
    --name yolov7_w01_logitKD_ver0_SCKD_8L \
    --logit-kd-loss V0 \
    --feature-kd-loss SCKD \
    --aug-off-epoch 15 \
    --hook-layer 11 24 37 50 63 75 88 101
}
yolo
