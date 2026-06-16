# -*- coding: utf-8 -*-
"""
recipe.py
팀원 전처리/추론 레시피를 한 곳에 고정한다.
infer.py 가 이 함수들만 쓰면, 글루 결과가 팀원 결과와 일치한다.

출처(팀원 코드에서 확정):
  - 검출(module/cell): preprocess_ct.py + eval_battery.py
      crop = battery_outline bbox + pad(max 25, 5%)  → grayscale RGB
      tile = 250×250, overlap 0.20  (stride 200)
      predict imgsz: module 512 / cell 640,  conf 0.05
      집계 = 타일→슬라이스→배터리 OR (k=1: 발화 타일 1개↑면 불량)
  - swelling: preprocess_swelling.py + eval_swelling.py
      crop = battery_outline bbox + pad(같은 식)  → letterbox 224 (회색)
      배터리 판정 = swelling 슬라이스 비율 > 0.1
"""
from PIL import Image

# ---- 검출 타일/크롭 (preprocess_ct.py 와 동일) ----
DET_TILE     = 250
DET_OVERLAP  = 0.20
PAD_RATIO    = 0.05
PAD_MIN      = 25

# ---- 추론 운영값 (README_HANDOFF §3) ----
DET_CONF        = 0.05
IMGSZ_MODULE    = 512
IMGSZ_CELL      = 640
SWELL_IMGSZ     = 224
SWELL_BATT_THR  = 0.1     # 배터리 swelling 슬라이스 비율 임계값

# 형태별 검출 클래스 인덱스 (README_HANDOFF §2)
#   module(detect, nc=2): porosity=0, resin overflow=1
#   cell  (detect, nc=1): porosity=0  (resin 없음)
POROSITY_IDX = 0
RESIN_IDX    = 1


def poly_bbox(points):
    """[x0,y0,x1,y1,...] 평면 좌표 폴리곤 → 외접 bbox (x0,y0,x1,y1)."""
    xs, ys = points[0::2], points[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def pad_crop_box(box, W, H):
    """outline bbox + pad(max(25, 5%)), 이미지 경계로 clip. (preprocess_ct.pad_crop_box 동일)"""
    x0, y0, x1, y1 = box
    pad = max(PAD_MIN, int(min(x1 - x0, y1 - y0) * PAD_RATIO))
    return (max(0, int(x0 - pad)), max(0, int(y0 - pad)),
            min(W, int(x1 + pad)), min(H, int(y1 + pad)))


def crop_outline(img, outline, W=None, H=None):
    """원본 슬라이스(PIL)에서 battery_outline 기준으로 크롭 → grayscale.
    반환: 회색('L') 크롭 이미지. (검출/seg/swelling 공통 크롭)"""
    if img.mode != "L":
        img = img.convert("L")
    if W is None or H is None:
        W, H = img.size
    box = pad_crop_box(poly_bbox(outline), W, H)
    return img.crop(box)


def to_rgb(gray):
    """회색('L') → RGB (같은 채널 3배). 검출 모델 2D 입력 형식(preprocess_ct.load_depth_crop 2D 경로)."""
    return Image.merge("RGB", (gray, gray, gray))


def tile_starts(length, tile=DET_TILE, overlap=DET_OVERLAP):
    """preprocess_ct.tile_starts 동일: 마지막 타일이 끝에 붙도록 보정."""
    step = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, max(1, length - tile + 1), step))
    if not starts or starts[-1] != length - tile:
        starts.append(max(0, length - tile))
    return sorted(set(starts))


def det_tiles(crop_rgb):
    """검출용 250×250 타일 생성기. yield (tile_img, tx, ty).
    가장자리 타일은 검은색으로 패딩해 250 정사각 유지."""
    W, H = crop_rgb.size
    for ty in tile_starts(H):
        for tx in tile_starts(W):
            t = crop_rgb.crop((tx, ty, tx + DET_TILE, ty + DET_TILE))
            if t.size != (DET_TILE, DET_TILE):
                cv = Image.new("RGB", (DET_TILE, DET_TILE), (0, 0, 0))
                cv.paste(t, (0, 0))
                t = cv
            yield t, tx, ty


def letterbox(gray, size=SWELL_IMGSZ):
    """swelling용: 종횡비 유지 + 검은 패딩(왜곡 방지). preprocess_swelling.letterbox 동일.
    입력/출력 모두 회색('L')."""
    if gray.mode != "L":
        gray = gray.convert("L")
    w, h = gray.size
    s = min(size / w, size / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    im2 = gray.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(im2, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def imgsz_for(cell_type):
    return IMGSZ_CELL if cell_type == "cell" else IMGSZ_MODULE
