#!/usr/bin/env bash
# swelling 5-fold: fold별 전개→학습→배터리단위 평가. 정상 배터리 신뢰구간(specificity) 확보용.
# 각 fold 데이터(yolo_swell_kf0~4.tar)는 로컬에서 업로드돼 있어야 함.
cd /workspace/battery-ct-security
mkdir -p /workspace/backup
LOG=/workspace/backup/swell_kfold.log
echo "=== swelling 5-fold 시작 $(date) ===" > $LOG
for f in 0 1 2 3 4; do
  echo "=== fold $f 전개+학습 $(date) ===" >> $LOG
  rm -rf /dev/shm/yolo_swell_kf$f
  tar -xf /workspace/yolo_swell_kf$f.tar -C /dev/shm
  python vision/train_swelling.py --data /dev/shm/yolo_swell_kf$f/images --name swell_kf$f \
    --epochs 60 --imgsz 224 --batch 128 >> $LOG 2>&1
  echo "=== fold $f 평가 ===" >> $LOG
  python vision/eval_swelling.py --weights runs/classify/models/swell_kf$f/weights/best.pt \
    --data /dev/shm/yolo_swell_kf$f/images --imgsz 224 > /workspace/backup/eval_swell_kf$f.txt 2>&1
  cat /workspace/backup/eval_swell_kf$f.txt >> $LOG
done
echo "KFOLD_DONE $(date)" >> $LOG
# 5 fold 요약: 각 fold 슬라이스 recall/normal recall 한 줄씩
echo "=== 요약 ===" >> $LOG
grep -h "슬라이스" /workspace/backup/eval_swell_kf*.txt >> $LOG
