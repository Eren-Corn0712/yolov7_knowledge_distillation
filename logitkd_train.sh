#!/bin/bash

function yolo() {
  python3 logitkd_train.py \
    --device 3 \
    --workers 16 \
    --batch-size 128 \
    --epochs 300 \
    --data data/bdd100k.yaml \
    --img 640 640 \
    --cfg cfg/yolov7_10c/yolov7-w01-10c.yaml \
    --hyp data/hyp.scratch.v7.w0.1.10c.kd1.yaml \
    --teacher-weights runs/yolov7/yolov7_w_10_10c_e600/weights/best.pt \
    --weights 'yolov7.pt' \
    --project runs/yolov7_kd \
    --name yolov7_w01_logitKD_ver0_cond2 \
    --kd-loss V0 \
    --aug-off-epoch 15
}

yolo
