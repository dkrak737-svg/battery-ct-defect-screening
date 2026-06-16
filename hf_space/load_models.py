# -*- coding: utf-8 -*-
"""
load_models.py (HF Space 판)
가중치를 HF 모델 레포에서 받아(캐시) 한 번만 로드. 로컬 weights/ 폴백도 지원.

  - 환경변수 HF_MODEL_REPO 가 있으면 그 레포에서 hf_hub_download (예: "user/battery-ct-defect-models")
  - 없으면 GLUE_WEIGHTS(기본 ./weights) 에서 로드
"""
import os
from pathlib import Path

from ultralytics import YOLO
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "").strip()
HERE = Path(__file__).resolve().parent
LOCAL_WDIR = Path(os.environ.get("GLUE_WEIGHTS", HERE / "weights"))

WEIGHTS = {
    "module":   "module_r01c.pt",
    "cell":     "cell_r06.pt",
    "swelling": [f"swell_kf{i}.pt" for i in range(5)],
    "seg":      "porosity_best.pt",
}


def _resolve(fname):
    if HF_MODEL_REPO:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=HF_MODEL_REPO, filename=fname)
    return str(LOCAL_WDIR / fname)


def load_all():
    """{'module': YOLO, 'cell': YOLO, 'swelling': [YOLO×5], 'seg': YOLO}"""
    src = HF_MODEL_REPO or str(LOCAL_WDIR)
    print(f"[load] device={DEVICE}  weights from: {src}")
    models = {
        "module":   YOLO(_resolve(WEIGHTS["module"])),
        "cell":     YOLO(_resolve(WEIGHTS["cell"])),
        "swelling": [YOLO(_resolve(f)) for f in WEIGHTS["swelling"]],
        "seg":      YOLO(_resolve(WEIGHTS["seg"])),
    }
    print(f"[module] {models['module'].names} | [swelling folds] {len(models['swelling'])}")
    return models


if __name__ == "__main__":
    load_all()
