#!/bin/bash
# yolo11n-seg porosity 학습. tmux에서 실행 권장:
#   tmux new -d -s seg 'bash vision/train_seg.sh 2>&1 | tee /root/seg.log'
# CLAUDE.md: lr0=0.005, warmup5, degrees=7(90도 회전 금지), 좌우/상하 flip, scale, mosaic.
set -e
cd /workspace/battery-ct-security
MODEL=${1:-yolo11n-seg.pt}     # 인자로 yolo11s-seg.pt 주면 probe 가능

yolo segment train \
  model=$MODEL \
  data=data/yolo_seg/dataset.yaml \
  epochs=40 imgsz=1024 batch=64 \
  lr0=0.005 warmup_epochs=5 \
  degrees=7 translate=0.1 scale=0.5 fliplr=0.5 flipud=0.5 mosaic=1.0 \
  patience=10 \
  project=runs/seg name=porosity exist_ok=True
# 학습 끝나면 best.pt -> runs/seg/porosity/weights/best.pt
# 평가: python vision/eval_seg.py
