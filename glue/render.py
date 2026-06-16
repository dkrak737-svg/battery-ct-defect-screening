# -*- coding: utf-8 -*-
"""
render.py
판정 결과 → 검사관용 출력 (대표 슬라이스 오버레이 + 보고서 + RAG 조치 지침).

표시 규칙(밑에 깔린 모델이 달라서):
  - 다공성(porosity) : seg 마스크 → 빨강 반투명 (대표 슬라이스에만 lazy 계산)
  - 레진(resin)      : 검출 bbox → 빨강 사각형
  - 팽창(swelling)   : 위치 없음(분류기) → 표시 안 함

좌표계: 검출 박스/표시 이미지 모두 'recipe 크롭' 좌표계라 그대로 겹쳐진다.
"""
import os
import sys

import cv2
import numpy as np

import recipe
from decide import DEFECT_CANON
from data import open_slice
from seg_tiles import seg_porosity_mask

# generate_guidance.py(RAG) 위치를 레이아웃에 무관하게 탐색.
#   로컬: ../teammate/rag,  HF Space: ./rag,  env GLUE_RAG_DIR 우선.
def _find_rag_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("GLUE_RAG_DIR"),
             os.path.join(here, "rag"),
             os.path.join(here, "teammate", "rag"),
             os.path.join(os.path.dirname(here), "teammate", "rag")]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "generate_guidance.py")):
            return c
    return cands[-1]

_RAG_DIR = _find_rag_dir()
if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)


# ----------------------------------------------------------------------
# 대표 슬라이스 선택 + 크롭 이미지
# ----------------------------------------------------------------------
def _crop_rgb(slices, idx):
    """idx 슬라이스의 크롭 이미지(검출/표시 공통 좌표계) → RGB ndarray."""
    sl = next((s for s in slices if s["idx"] == idx), None)
    if sl is None:
        return None
    return np.array(recipe.to_rgb(open_slice(sl)))   # 이미 크롭됨


def pick_slice(result, slices):
    """대표 슬라이스 = 검출 신뢰도가 가장 높은 슬라이스.
    검출 박스가 없으면(예: swelling 단독) (None, None) → 위치 표시 생략.
    반환: (slice_idx, crop_rgb ndarray)."""
    best_idx, best_conf = None, -1.0
    for box in result["porosity"]["boxes"] + result["resin"]["boxes"]:
        cf, si = box[4], int(box[5])
        if cf > best_conf:
            best_conf, best_idx = cf, si
    if best_idx is None:
        return None, None
    return best_idx, _crop_rgb(slices, best_idx)


def _blend_mask(img, mask, color=(255, 0, 0), alpha=0.45):
    """마스크 영역 반투명 색(RGB)."""
    overlay = img.copy()
    overlay[mask.astype(bool)] = color
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def make_overlay(crop_rgb, result, cur, seg_model=None):
    """대표 슬라이스(cur) 위에 그 슬라이스의 결함만 그린다."""
    if crop_rgb is None:
        return None
    img = crop_rgb.copy()

    # 다공성 → seg 마스크(이 크롭에만 lazy). 실패 시 검출 박스로 폴백.
    drew_mask = False
    if seg_model is not None and result["porosity"]["flag"]:
        try:
            mask, _ = seg_porosity_mask(seg_model, crop_rgb, result["cell_type"])
            if mask.any():
                img = _blend_mask(img, mask)
                drew_mask = True
        except Exception as e:
            print(f"[seg 생략] {e}", file=sys.stderr)
    if not drew_mask:
        for x1, y1, x2, y2, cf, si in result["porosity"]["boxes"]:
            if si == cur:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # 레진 → 박스
    for x1, y1, x2, y2, cf, si in result["resin"]["boxes"]:
        if si == cur:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 128, 0), 2)
    # 팽창 → 표시 없음
    return img


# ----------------------------------------------------------------------
# 보고서 + RAG 조치 지침
# ----------------------------------------------------------------------
def make_report(result, decision):
    sw = result["swelling"]
    return {
        "배터리":     result["battery_id"],
        "셀 형태":    result["cell_type"],
        "판정":       decision["zone"],
        "결함 종류":  [DEFECT_CANON[d] for d in decision["defects"]] or ["없음"],
        "최고 신뢰도": round(decision["worst_conf"], 3),
        "swelling 비율": f"{sw['ratio']*100:.0f}% ({sw['n_swell']}/{result['n_slices']})",
        "사유":       decision["reasons"],
    }


def recommend(result, decision):
    """RAG(generate_guidance) → IATA 규정 기반 조치 지침(한/영).
    결함 없으면 '이상없음' 표기(API 호출 없음). 키/네트워크 없으면 결정적 매핑으로 폴백."""
    import generate_guidance as gg

    defects = [DEFECT_CANON[d] for d in decision["defects"]]
    if not defects:
        return f"{gg.NORMAL_KO}\n\n{gg.NORMAL_EN}"

    battery = {"battery_id": result["battery_id"],
               "form": result["cell_type"], "defects": defects}
    try:
        kb, by_id = gg.load_kb()
        chunks = gg.retrieve(defects, kb, by_id)
        return gg.generate(battery, chunks)          # Claude API (ANTHROPIC_API_KEY 필요)
    except Exception as e:
        # 폴백: 결정적 매핑만으로 근거 조항 나열 (API 불가 시 데모 지속)
        try:
            kb, by_id = gg.load_kb()
            mapping = kb["meta"]["defect_to_chunk"]
            refs = []
            for d in defects:
                for cid in mapping.get(d, []):
                    c = by_id.get(cid)
                    if c:
                        refs.append(f"- [{c['source']} · {c['ref']}] {c['text_ko']}")
            body = "\n".join(dict.fromkeys(refs)) or "(매핑 조항 없음)"
            return (f"⚠️ AI 지침 생성 불가({e}) → 규정 매핑만 표시.\n"
                    f"탐지 결함: {', '.join(defects)}\n{body}")
        except Exception:
            return f"⚠️ 지침 생성 불가: {e}"
