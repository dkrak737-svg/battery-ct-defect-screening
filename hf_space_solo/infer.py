# -*- coding: utf-8 -*-
"""
infer.py (solo)
본인 모델만으로 배터리 추론 → 표준 결과(decide/render 공용 형식).
  - swelling: timm 분류기, 슬라이스 P(swell)≥0.5, 배터리 k=1
  - porosity: YOLO seg, 슬라이스별 마스크. 대표(최고 conf) 슬라이스 마스크 보관(오버레이용)
  - resin 트랙 없음(본인 모델 아님) → flag False 로 자리만(decide 호환)
"""
import recipe
from data import open_slice
from seg_tiles import seg_porosity_mask


def infer_battery(models, slices, cell_type, battery_id):
    grays = [open_slice(sl) for sl in slices]

    # 1) swelling (timm)
    probs = models["swell"].prob_swell(grays)
    n_swell = int((probs >= 0.5).sum())
    swelling = {
        "flag": n_swell >= 1,                       # 운영 k=1
        "ratio": n_swell / max(1, len(slices)),
        "n_swell": n_swell,
        "conf": float(probs.max()) if len(probs) else 0.0,
    }

    # 2) porosity (seg) — 슬라이스별, 대표 마스크 보관
    best_conf, best = 0.0, None
    hit = 0
    for sl, g in zip(slices, grays):
        mask, conf = seg_porosity_mask(models["seg"], recipe.to_rgb(g), cell_type)
        if mask.any():
            hit += 1
            if conf > best_conf:
                best_conf, best = conf, {"slice": sl["idx"], "mask": mask}
    porosity = {
        "flag": hit > 0,
        "conf": best_conf,
        "boxes": [],
        "best_slice": best["slice"] if best else None,
        "best_mask": best["mask"] if best else None,
    }

    return {
        "battery_id": battery_id,
        "cell_type":  cell_type,
        "n_slices":   len(slices),
        "swelling":   swelling,
        "porosity":   porosity,
        "resin":      {"flag": False, "conf": 0.0, "boxes": []},
    }
