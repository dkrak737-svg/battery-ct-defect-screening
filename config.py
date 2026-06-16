"""Centralized config for battery CT detection.

Works on both local Windows (d:/CT DATA) and RunPod Linux (/workspace).
Auto-detects environment via RUNPOD_POD_ID env var.
"""
import os
from pathlib import Path

# ---------- Environment ----------
ON_RUNPOD = (os.environ.get('RUNPOD_POD_ID') is not None
             or Path('/workspace').exists())

if ON_RUNPOD:
    DATA_ROOT = Path('/workspace/battery-ct-security/data')
else:
    DATA_ROOT = Path('d:/CT DATA')

# ---------- Source paths (raw AI-Hub data) ----------
TRAIN_IMG_DIR   = DATA_ROOT / 'Training'   / '원천데이터'
TRAIN_LABEL_DIR = DATA_ROOT / 'Training'   / '라벨링데이터'
VAL_IMG_DIR     = DATA_ROOT / 'Validation' / '원천데이터'
VAL_LABEL_DIR   = DATA_ROOT / 'Validation' / '라벨링데이터'

# ---------- Preprocessed outputs (this is what gets uploaded to RunPod) ----------
CROPPED_DIR  = DATA_ROOT / 'cropped'
LABELS_JSONL = DATA_ROOT / 'labels.jsonl'

# ---------- Classes ----------
CLASSES = ['porosity', 'resin_overflow', 'swelling']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# Map raw JSON `defects[].name` values to our class names.
# (AI-Hub labels use space "resin overflow"; we normalize to underscore.)
LABEL_NAME_TO_CLASS = {
    'porosity'      : 'porosity',
    'resin overflow': 'resin_overflow',
    'resin_overflow': 'resin_overflow',  # safety
}


def get_class_id(raw_name: str):
    """JSON 'name' -> internal class id (or None if unknown)."""
    if raw_name is None:
        return None
    canon = LABEL_NAME_TO_CLASS.get(raw_name.strip())
    if canon is None:
        return None
    return CLASS_TO_ID[canon]

# ---------- Crop config (LOCAL preprocessing — fixed rules) ----------
CROP_PADDING_PX  = 50
MIN_BBOX_SIZE_PX = 2    # bboxes smaller than this are dropped at conversion
JPG_QUALITY      = 90   # cropped image save quality

# ---------- Tile config (RunPod tiling — tuning target) ----------
TILE_SIZE      = 1024
TILE_OVERLAP   = 0.25
TILE_STRIDE    = int(TILE_SIZE * (1 - TILE_OVERLAP))  # 768
MIN_TILE_BBOX_PX = 4    # after tile clipping, smaller bboxes are dropped

# ---------- Split config ----------
SPLIT_RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}
RANDOM_SEED  = 42

# ---------- Multiprocessing ----------
NUM_WORKERS = 8

# ---------- Tile augmentation (r02 — oversample rare classes) ----------
# duplicate tiles containing this many copies if they have resin_overflow boxes
RESIN_OVERFLOW_OVERSAMPLE = 10  # r01: 1, r02: 10x — rare class boost

# ---------- Training (r02 settings — see r01_분석보고서.md) ----------
YOLO_MODEL_BASE = 'yolo11s.pt'  # r01: n (2.6M) -> r02: s (9.4M) — capacity up
EPOCHS  = 40                     # r01: 30 -> r02: 40
IMG_SZ  = 1024
BATCH   = 64                     # r01: 96 -> r02: 64 — stability up
LR0     = 0.005                  # r01: 0.01 -> r02: 0.005 — prevent epoch-4 catastrophic drop
WARMUP_EPOCHS = 5                # r01: 3 -> r02: 5 — gentler warmup
AMP     = False                  # r01: True -> r02: False — kill box_loss=inf episodes
PATIENCE = 10                    # r01: 15 -> r02: 10 — stop earlier when plateau hits
