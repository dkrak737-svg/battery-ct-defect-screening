"""배터리 단위 K-fold 분할 (데이터 재생성 없이 fold txt/yaml만 생성).

현재 /dev/shm/yolo_cell 의 train+val 타일(전체 47배터리)을 배터리 단위로 K등분.
각 fold: 해당 배터리들=val, 나머지=train. ultralytics 는 train/val 에 이미지경로 txt 허용.
보고서용: 배터리 단위 recall 신뢰구간 산출.
"""
import glob
import os
import re

ROOT = "/dev/shm/yolo_cell"
K = 3
SEED = 42

imgs = glob.glob(f"{ROOT}/images/train/*.jpg") + glob.glob(f"{ROOT}/images/val/*.jpg")
bat = {}
for p in imgs:
    m = re.match(r"(CT_cell_pouch_\d+)", os.path.basename(p))
    bat.setdefault(m.group(1), []).append(p)

bats = sorted(bat)
# 시드 고정 셔플(Random 모듈, deterministic)
import random
random.Random(SEED).shuffle(bats)
folds = [bats[i::K] for i in range(K)]  # stride 분할(균등)

print(f"전체 배터리 {len(bats)}개 -> {K}-fold: {[len(f) for f in folds]}")
for i in range(K):
    val_bats = set(folds[i])
    val_imgs = [p for b in folds[i] for p in bat[b]]
    train_imgs = [p for b in bats if b not in val_bats for p in bat[b]]
    open(f"{ROOT}/val_f{i}.txt", "w").write("\n".join(val_imgs) + "\n")
    open(f"{ROOT}/train_f{i}.txt", "w").write("\n".join(train_imgs) + "\n")
    open(f"{ROOT}/fold{i}.yaml", "w").write(
        f"path: {ROOT}\ntrain: train_f{i}.txt\nval: val_f{i}.txt\nnc: 1\nnames: ['porosity']\n")
    print(f"  fold{i}: val 배터리 {len(val_bats)}개 / val타일 {len(val_imgs)} / train타일 {len(train_imgs)}")
print("생성: fold0/1/2.yaml + train_f*/val_f*.txt")
