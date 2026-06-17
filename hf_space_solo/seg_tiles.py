# -*- coding: utf-8 -*-
"""
seg_tiles.py
porosity 세그멘테이션(내 모델 porosity_best.pt)용 타일링 + 마스크 stitch.
tile_seg.py(학습용) 의 타일 로직을 추론용으로 압축해 재사용한다.

레시피(tile_seg.py 와 동일):
  - cell 은 등방 4배 확대(얇은 porosity 살림, 모양 보존), module 은 1배
  - 1024×1024 타일, overlap 0.25 (stride 768), 가장자리 검은 패딩
출력: 표시용 크롭과 같은 좌표계의 이진 마스크 1장(+ 최대 conf).

⚠️ seg 는 '정밀 위치 오버레이' 용도라 대표 슬라이스 1장에만 돌린다(전 슬라이스 X → 데모 속도).
"""
import numpy as np
from PIL import Image

SEG_TILE   = 1024
SEG_OVERLAP = 0.25
CELL_SCALE = 4


def _origins(size, tile, stride):
    if size <= tile:
        return [0]
    xs = list(range(0, size - tile + 1, stride))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return xs


def seg_porosity_mask(model, crop_gray, cell_type, conf=0.10):
    """표시용 크롭(회색 PIL 또는 ndarray)에 porosity 마스크를 만든다.
    반환: (mask[H,W] bool — 크롭 좌표계, max_conf)."""
    import cv2
    if isinstance(crop_gray, np.ndarray):
        crop_gray = Image.fromarray(crop_gray)
    if crop_gray.mode != "RGB":
        crop_gray = crop_gray.convert("RGB")

    W0, H0 = crop_gray.size
    scale = CELL_SCALE if cell_type == "cell" else 1
    im = crop_gray.resize((W0 * scale, H0 * scale)) if scale != 1 else crop_gray
    W, H = im.size
    stride = int(SEG_TILE * (1 - SEG_OVERLAP))

    mask = np.zeros((H0, W0), dtype=np.uint8)
    best = 0.0
    for oy in _origins(H, SEG_TILE, stride):
        for ox in _origins(W, SEG_TILE, stride):
            tile = im.crop((ox, oy, ox + SEG_TILE, oy + SEG_TILE))
            if tile.size != (SEG_TILE, SEG_TILE):
                cv = Image.new("RGB", (SEG_TILE, SEG_TILE), (0, 0, 0))
                cv.paste(tile, (0, 0))
                tile = cv
            r = model.predict(tile, conf=conf, verbose=False)[0]
            if r.masks is None:
                continue
            for poly, b in zip(r.masks.xy, r.boxes):
                cf = float(b.conf)
                best = max(best, cf)
                pts = np.array(poly, dtype=np.float32)
                if len(pts) < 3:
                    continue
                # 타일 좌표 → 확대 크롭 좌표 → 원본 크롭 좌표
                pts[:, 0] = (pts[:, 0] + ox) / scale
                pts[:, 1] = (pts[:, 1] + oy) / scale
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool), best
