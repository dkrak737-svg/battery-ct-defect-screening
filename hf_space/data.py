# -*- coding: utf-8 -*-
"""
data.py
이미 크롭된 슬라이스(cropped/ + labels.jsonl)에서 배터리를 읽어온다.

이 팟의 데이터는 원본이 아니라 '내 전처리 출력'이다:
  data/cropped/<battery_key>/<slice>.jpg   (배터리 outline+여유로 크롭된 회색 슬라이스)
  data/labels.jsonl   (슬라이스별: set/path/battery_key/type(cell|module)/swelling_flag/boxes(크롭좌표 GT))
→ 이미 크롭돼 있으므로 추론은 outline 크롭 없이 '크롭 이미지 그대로' 타일링/letterbox 한다.

루트 지정(우선순위): GLUE_DATA_ROOT > /workspace/battery-ct-security/data > d:/CT DATA
"""
import json
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image

# 배터리당 슬라이스 상한(데모 속도; 축당 균등 샘플). 0 이면 전량. env GLUE_MAX_SLICES.
MAX_SLICES = int(os.environ.get("GLUE_MAX_SLICES", "0"))

_ROOT = None
_INDEX = None     # {battery_key: {"type": str, "slices": [(name, rel_path)]}}


def _candidates():
    env = os.environ.get("GLUE_DATA_ROOT")
    if env:
        yield Path(env)
    yield Path("/workspace/battery-ct-security/data")
    yield Path("d:/CT DATA")


def data_root():
    global _ROOT
    if _ROOT is not None:
        return _ROOT
    for c in _candidates():
        if c and (c / "labels.jsonl").exists():
            _ROOT = c
            return _ROOT
    raise FileNotFoundError(
        "labels.jsonl 을 못 찾음. GLUE_DATA_ROOT 로 data 폴더를 지정하세요.")


def _build_index():
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    root = data_root()
    idx = defaultdict(lambda: {"type": None, "slices": []})
    with open(root / "labels.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            bk = r["battery_key"]
            idx[bk]["type"] = r.get("type")
            idx[bk]["slices"].append((Path(r["path"]).stem, r["path"]))
    for bk in idx:
        idx[bk]["slices"].sort(key=lambda s: s[0])
    _INDEX = idx
    return idx


def list_batteries():
    return sorted(_build_index().keys())


def cell_type_of(bid):
    t = _build_index().get(bid, {}).get("type")
    return t if t in ("cell", "module") else ("cell" if "cell" in bid else "module")


def _subsample(records, k):
    if k <= 0 or len(records) <= k:
        return records
    return [records[round(t * (len(records) - 1) / (k - 1))] for t in range(k)]


def load_battery(bid):
    """배터리 하나 → (slices, cell_type).
    slices = [{"idx", "name", "img_path"(크롭 이미지 절대경로)}]  (이미 크롭됨 → outline 불필요)."""
    idx = _build_index()
    if bid not in idx:
        raise KeyError(f"배터리 없음: {bid}")
    root = data_root()
    recs = _subsample(idx[bid]["slices"], MAX_SLICES)
    slices = [{"idx": i, "name": name, "img_path": str(root / "cropped" / rel)}
              for i, (name, rel) in enumerate(recs)]
    return slices, cell_type_of(bid)


def open_slice(sl):
    """크롭 슬라이스 → 회색 이미지(PIL 'L'). (이미 크롭된 이미지 그대로)."""
    return Image.open(sl["img_path"]).convert("L")
