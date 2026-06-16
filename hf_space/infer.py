# -*- coding: utf-8 -*-
"""
infer.py
배터리 하나(슬라이스 묶음)를 모델에 돌려 '표준 결과(BatteryResult)'를 만든다.
전처리/집계는 recipe.py(=팀원 레시피)만 사용 → 결과가 팀원과 일치.

구성(실제 받은 자료 기준):
  - swelling : module 전용, 5-fold 앙상블 평균 → 슬라이스 swelling 비율 > 0.1 이면 배터리 swelling
  - 검출     : 형태별 모델(module=porosity+resin / cell=porosity), 250 타일 OR 집계
  - seg(정밀 마스크)는 여기서 안 돌린다 → render 에서 대표 슬라이스 1장에만(속도).
"""
from typing import TypedDict, List

import numpy as np

import recipe
from data import open_slice


class BatteryResult(TypedDict):
    battery_id: str
    cell_type: str                 # "module" | "cell"
    n_slices: int
    swelling: dict                 # {"flag", "ratio", "n_swell", "conf"}            (위치 없음)
    porosity: dict                 # {"flag", "conf", "boxes"}  box=[x1,y1,x2,y2,conf,slice_idx] (크롭좌표)
    resin:    dict                 # {"flag", "conf", "boxes"}


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ----------------------------------------------------------------------
# 1) swelling — module 전용, 5-fold 앙상블
# ----------------------------------------------------------------------
def run_swelling(folds, slices, cell_type) -> dict:
    """5-fold 확률 평균 → 슬라이스별 top1 → 배터리 swelling 비율로 판정.
    cell 은 swelling 트랙 없음 → flag False."""
    empty = {"flag": False, "ratio": 0.0, "n_swell": 0, "conf": 0.0}
    if cell_type != "module" or not slices:
        return empty

    # 이미 크롭된 슬라이스 → letterbox 224 (회색). 한 번만 만들어 5 fold 에 재사용.
    crops = []
    for sl in slices:
        crops.append(recipe.letterbox(open_slice(sl), recipe.SWELL_IMGSZ))

    swell_idx = [k for k, v in folds[0].names.items() if v == "swelling"][0]
    sum_probs = np.zeros((len(crops), len(folds[0].names)), dtype=np.float64)
    for fold in folds:
        row = 0
        for ch in _chunks(crops, 256):
            res = fold.predict(ch, imgsz=recipe.SWELL_IMGSZ, verbose=False)
            for r in res:
                sum_probs[row] += r.probs.data.cpu().numpy()
                row += 1
    avg = sum_probs / len(folds)

    pred = avg.argmax(axis=1)
    is_swell = pred == swell_idx
    n_swell = int(is_swell.sum())
    ratio = n_swell / len(crops)
    if n_swell:
        conf = float(avg[is_swell, swell_idx].mean())     # swelling 판정 슬라이스들의 평균 확신도
    else:
        conf = float(avg[:, swell_idx].max())
    return {"flag": ratio > recipe.SWELL_BATT_THR, "ratio": ratio,
            "n_swell": n_swell, "conf": conf}


# ----------------------------------------------------------------------
# 2) 검출 — 형태별 모델, 250 타일 OR 집계
# ----------------------------------------------------------------------
def run_detection(det_model, slices, cell_type, conf=None) -> dict:
    """타일 박스를 크롭 좌표로 모으고 OR 집계. cell 모델은 porosity(0)만, resin 없음.
    box = [x1, y1, x2, y2, conf, slice_idx]  (크롭 좌표계)."""
    conf = recipe.DET_CONF if conf is None else conf
    imgsz = recipe.imgsz_for(cell_type)
    has_resin = (cell_type == "module")     # cell 모델은 nc=1

    por_boxes, res_boxes = [], []
    por_conf, res_conf = 0.0, 0.0

    for sl in slices:
        rgb = recipe.to_rgb(open_slice(sl))      # 이미 크롭된 슬라이스
        tiles, offs = [], []
        for t, tx, ty in recipe.det_tiles(rgb):
            tiles.append(t); offs.append((tx, ty))
        if not tiles:
            continue
        for ch_t, ch_o in zip(_chunks(tiles, 64), _chunks(offs, 64)):
            results = det_model.predict(ch_t, conf=conf, imgsz=imgsz, verbose=False)
            for r, (tx, ty) in zip(results, ch_o):
                for b in r.boxes:
                    c, cf = int(b.cls), float(b.conf)
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    box = [x1 + tx, y1 + ty, x2 + tx, y2 + ty, cf, sl["idx"]]
                    if c == recipe.POROSITY_IDX:
                        por_boxes.append(box); por_conf = max(por_conf, cf)
                    elif has_resin and c == recipe.RESIN_IDX:
                        res_boxes.append(box); res_conf = max(res_conf, cf)

    return {
        "porosity": {"flag": len(por_boxes) > 0, "conf": por_conf, "boxes": por_boxes},
        "resin":    {"flag": len(res_boxes) > 0, "conf": res_conf, "boxes": res_boxes},
    }


# ----------------------------------------------------------------------
# 묶기 — 표준 결과 반환 (seg 마스크는 render 가 대표 슬라이스에만 lazy 계산)
# ----------------------------------------------------------------------
def infer_battery(models, slices, cell_type, battery_id) -> BatteryResult:
    sw = run_swelling(models["swelling"], slices, cell_type)
    det_model = models["module"] if cell_type == "module" else models["cell"]
    det = run_detection(det_model, slices, cell_type)

    return {
        "battery_id": battery_id,
        "cell_type":  cell_type,
        "n_slices":   len(slices),
        "swelling":   sw,
        "porosity":   det["porosity"],
        "resin":      det["resin"],
    }
