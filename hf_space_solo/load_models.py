# -*- coding: utf-8 -*-
"""
load_models.py (solo Space)
본인 모델 2개만 로드:
  - swelling: timm efficientnet_b0 (swelling_best.pt)
  - porosity: YOLO11-seg (porosity_best.pt)
HF_MODEL_REPO 가 있으면 거기서 받고(캐시), 없으면 로컬 weights/.
"""
import os
from pathlib import Path

import torch

from swelling_timm import SwellingClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "").strip()
HERE = Path(__file__).resolve().parent
LOCAL_WDIR = Path(os.environ.get("GLUE_WEIGHTS", HERE / "weights"))


def _resolve(fname):
    if HF_MODEL_REPO:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=HF_MODEL_REPO, filename=fname)
    return str(LOCAL_WDIR / fname)


def load_all():
    """{'swell': SwellingClassifier, 'seg': YOLO}"""
    from ultralytics import YOLO
    src = HF_MODEL_REPO or str(LOCAL_WDIR)
    print(f"[load·solo] device={DEVICE} weights from: {src}")
    models = {
        "swell": SwellingClassifier(_resolve("swelling_best.pt"), imgsz=512),
        "seg":   YOLO(_resolve("porosity_best.pt")),
    }
    print("[swell] timm efficientnet_b0 (nonswell=0/swell=1) | [seg]", models["seg"].names)
    return models


if __name__ == "__main__":
    load_all()
