# -*- coding: utf-8 -*-
"""
load_models.py
모델을 한 번만 로드해서 재사용한다 (앱이 요청마다 다시 읽지 않도록).

실제 받은 자료 기준(README_HANDOFF):
  - module 검출  : module_r01c.pt  (detect, nc=2: porosity, resin overflow)  ← 팀원
  - cell 검출    : cell_r06.pt      (detect, nc=1: porosity)                  ← 팀원
  - swelling     : swell_kf0~4.pt   (classify, 5-fold 앙상블; normal=0/swelling=1) ← 팀원
  - porosity seg : porosity_best.pt (segment, 정밀 위치 오버레이용)            ← 너 (루트)
"""
import os
from pathlib import Path

from ultralytics import YOLO
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HERE = Path(__file__).resolve().parent
# 가중치 폴더: env GLUE_WEIGHTS > glue/weights. (배포 시 7종을 한 폴더에 모음)
WDIR = Path(os.environ.get("GLUE_WEIGHTS", HERE / "weights"))

PATHS = {
    "module":   WDIR / "module_r01c.pt",
    "cell":     WDIR / "cell_r06.pt",
    "swelling": [WDIR / f"swell_kf{i}.pt" for i in range(5)],   # 5-fold 앙상블
    "seg":      WDIR / "porosity_best.pt",
}


def load_all(paths=PATHS):
    """모델 모두 로드해서 dict 반환.
    {'module': YOLO, 'cell': YOLO, 'swelling': [YOLO×5], 'seg': YOLO}"""
    models = {
        "module":   YOLO(str(paths["module"])),
        "cell":     YOLO(str(paths["cell"])),
        "swelling": [YOLO(str(p)) for p in paths["swelling"]],
        "seg":      YOLO(str(paths["seg"])),
    }

    # 점검 출력 — task 와 클래스 순서 확인 (recipe.py 의 *_IDX 와 일치해야 함)
    print(f"[device] {DEVICE}")
    print(f"[module] task={models['module'].task}  names={models['module'].names}")
    print(f"[cell]   task={models['cell'].task}  names={models['cell'].names}")
    print(f"[swelling] folds={len(models['swelling'])}  names={models['swelling'][0].names}")
    print(f"[seg]    task={models['seg'].task}  names={models['seg'].names}")
    return models


if __name__ == "__main__":
    load_all()   # 단독 실행: 모델 열림 + 클래스 순서 확인용
