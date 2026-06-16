"""swelling 이진 분류 데이터셋 (module). YOLO11-cls 폴더 구조.
  out/images/train/{normal,swelling}/*.jpg, out/images/val/{normal,swelling}/*.jpg

설계:
  - 라벨 = 슬라이스의 swelling 필드(true=swelling, false=normal). swelling은 전역(배터리 전체 부풂).
  - 분할 = 배터리(form_num) 단위(누수 방지). swelling 배터리 67 / 정상 20.
  - 배터리당 균등 샘플(--per-battery)로 11만 슬라이스를 학습 가능 규모로 축소.
  - 2단계: ①정규식으로 swelling 필드만 빠르게 → 샘플 선정 ②선정분만 크롭/리사이즈(I/O 절약).
  - 크롭 = battery_outline bbox(+패딩), resize 시 종횡비 유지 + 패딩(외형이 핵심이라 왜곡 방지).
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

SEED = 42
NAME_PAT = re.compile(r"CT_module_pouch_(\d+)_[xyz]_\d+")
SWELL_TRUE = re.compile(r'"swelling"\s*:\s*true')
PAD_RATIO = 0.05
PAD_MIN = 25


def battery_key(name):
    m = NAME_PAT.search(name)
    return f"module_pouch_{m.group(1)}" if m else None


def poly_bbox(pts):
    xs, ys = pts[0::2], pts[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def letterbox(im, size):
    """종횡비 유지 + 패딩(배터리 외형 왜곡 방지)."""
    w, h = im.size
    s = min(size / w, size / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    im2 = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(im2, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-swelling", type=int, default=60, help="swelling 배터리당 샘플 슬라이스")
    ap.add_argument("--per-normal", type=int, default=200, help="정상 배터리당 샘플 슬라이스")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--n-folds", type=int, default=1, help="k-fold 수(1=단일 split)")
    ap.add_argument("--fold-idx", type=int, default=0, help="이 fold를 val로(0~n_folds-1)")
    args = ap.parse_args()
    random.seed(SEED)

    # os.walk 한 번으로 jpg/json 동시 수집 (rglob 2회보다 훨씬 빠름)
    imgmap, jpaths = {}, []
    for root, _, files in os.walk(args.src):
        for f in files:
            if not f.startswith("CT_module_pouch_"):
                continue
            if f.endswith(".jpg"):
                imgmap[f[:-4]] = os.path.join(root, f)
            elif f.endswith(".json"):
                jpaths.append(os.path.join(root, f))
    print(f"module jpg {len(imgmap)}개, json {len(jpaths)}개")

    # pass1: json 정규식으로 (배터리 -> swelling/normal 슬라이스 stem 리스트)
    by_bat = defaultdict(lambda: {"swelling": [], "normal": []})
    jsonmap = {}
    n = 0
    for jp in jpaths:
        stem = os.path.splitext(os.path.basename(jp))[0]
        bat = battery_key(stem)
        if not bat or stem not in imgmap:
            continue
        jsonmap[stem] = jp
        try:
            txt = open(jp, encoding="utf-8").read()
        except Exception:
            continue
        cls = "swelling" if SWELL_TRUE.search(txt) else "normal"
        by_bat[bat][cls].append(stem)
        n += 1
        if n % 30000 == 0:
            print(f"  ...pass1 {n} json")
    print(f"pass1 완료: 배터리 {len(by_bat)}개")

    # 배터리 단위 분할 (swelling 보유 여부로 stratify)
    swell_bats = sorted(b for b in by_bat if by_bat[b]["swelling"])
    norm_bats = sorted(b for b in by_bat if not by_bat[b]["swelling"])
    random.shuffle(swell_bats); random.shuffle(norm_bats)
    if args.n_folds > 1:
        # k-fold: 셔플 후 stride 로 fold_idx 번째를 val (정상/swelling 각각 균등 분산)
        val_s = swell_bats[args.fold_idx::args.n_folds]
        val_n = norm_bats[args.fold_idx::args.n_folds]
        val_bats = set(val_s + val_n)
        print(f"[{args.n_folds}-fold, fold {args.fold_idx}] swelling val {len(val_s)}/{len(swell_bats)}, 정상 val {len(val_n)}/{len(norm_bats)}")
    else:
        nv_s = max(1, int(len(swell_bats) * args.val_ratio))
        nv_n = max(1, int(len(norm_bats) * args.val_ratio))
        val_bats = set(swell_bats[:nv_s] + norm_bats[:nv_n])
        print(f"swelling 배터리 {len(swell_bats)}(val {nv_s}) / 정상 배터리 {len(norm_bats)}(val {nv_n})")

    for split in ("train", "val"):
        for c in ("normal", "swelling"):
            (Path(args.out) / "images" / split / c).mkdir(parents=True, exist_ok=True)

    def sample(lst, k):
        if len(lst) <= k:
            return lst
        return [lst[round(t * (len(lst) - 1) / (k - 1))] for t in range(k)]

    # pass2: 선정 슬라이스만 크롭/리사이즈
    # ★ 라벨 정제: swelling 배터리는 swelling 단면만, 정상 배터리는 normal 단면만 사용
    #   (swelling 배터리의 애매한 '정상 단면'은 제외 -> 깨끗한 배터리단위 라벨)
    stats = defaultdict(int)
    for bat, d in by_bat.items():
        split = "val" if bat in val_bats else "train"
        if d["swelling"]:  # swelling 배터리
            plan = [("swelling", sorted(d["swelling"]), args.per_swelling)]
        else:              # 정상 배터리
            plan = [("normal", sorted(d["normal"]), args.per_normal)]
        for cls, stems, k in plan:
            for stem in sample(stems, k):
                try:
                    j = json.loads(open(jsonmap[stem], encoding="utf-8").read())
                    outline = (j.get("swelling") or {}).get("battery_outline")
                    if not outline:
                        stats["no_outline"] += 1
                        continue
                    x0, y0, x1, y1 = poly_bbox(outline)
                    pad = max(PAD_MIN, int(min(x1 - x0, y1 - y0) * PAD_RATIO))
                    im = Image.open(imgmap[stem]).convert("L")
                    W, H = im.size
                    crop = im.crop((max(0, int(x0 - pad)), max(0, int(y0 - pad)),
                                    min(W, int(x1 + pad)), min(H, int(y1 + pad))))
                    out = letterbox(crop, args.imgsz)
                    out.save(Path(args.out) / "images" / split / cls / f"{stem}.jpg", quality=90)
                    stats[f"{split}_{cls}"] += 1
                except Exception as e:
                    stats["err"] += 1

    print("\n=== 통계 ===")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"  데이터: {args.out}/images/{{train,val}}/{{normal,swelling}}")


if __name__ == "__main__":
    main()
