# -*- coding: utf-8 -*-
"""
render_solo.py
solo 결과용 오버레이. 보고서·RAG 지침은 render.py 것을 재사용.
porosity = seg 마스크(빨강 반투명), swelling = 위치 없음.
"""
import cv2
import numpy as np

import recipe
from data import open_slice
from render import make_report, recommend   # 재사용


def _crop_rgb(slices, idx):
    sl = next((s for s in slices if s["idx"] == idx), None)
    if sl is None:
        return None
    return np.array(recipe.to_rgb(open_slice(sl)))


def overlay(result, slices):
    """대표(최고 conf) porosity 슬라이스에 마스크 오버레이. 없으면 None."""
    por = result["porosity"]
    if not por.get("flag") or por.get("best_slice") is None:
        return None
    crop = _crop_rgb(slices, por["best_slice"])
    if crop is None:
        return None
    mask = por["best_mask"]
    painted = crop.copy()
    painted[mask.astype(bool)] = (255, 0, 0)
    return cv2.addWeighted(painted, 0.45, crop, 0.55, 0)
