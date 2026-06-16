"""
배터리 CT 전처리: 크롭 + 폴리곤->bbox + 타일링 -> YOLO 데이터셋 (형태별 모델용)

설계(전체 데이터 스캔 반영):
  - 형태(form) 2종을 분리 학습: cell_pouch(1:13 막대) / module_pouch(1:2.4 직사각).
    크롭 종횡비가 완전히 달라 모델을 나눈다. --form 으로 한쪽만 처리.
  - 클래스: cell -> {porosity}, module -> {porosity, resin overflow}.
  - 분할: AI-Hub 기본 train/val 은 같은 배터리를 슬라이스로 쪼개 누수 -> 무시.
    train+val 폴더를 모두 긁어 "form_num" 배터리 단위로 재분할.
  - 결함 중심 축소:
      * defects 있는 슬라이스 = positive 소스 (전부 사용)
      * is_normal=true 완전 정상 슬라이스 = negative 소스 (--normal-sample-ratio 로 축소)
      * swelling=true & defects=null = 스웰링 전용 -> detection 에서 제외(별도 트랙)
  - 타일링: 250x250, ~20% 겹침. 채택 기준 = (결함대비 비율) OR (타일대비 비율).

좌표계: defects.points / battery_outline 모두 원본 4000x4000 (viz_check.py 검증).

사용:
  # cell 소규모 테스트 (Sample)
  python vision/preprocess_ct.py --src "d:/shinbaram/데이터/Sample/Sample" --form cell_pouch --out "d:/shinbaram/data/yolo_cell" --viz
  # 전체 cell
  python vision/preprocess_ct.py --src "d:/shinbaram/데이터/103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터" --form cell_pouch --out "d:/shinbaram/data/yolo_cell"
  # 전체 module
  python vision/preprocess_ct.py --src "d:/shinbaram/데이터/103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터" --form module_pouch --out "d:/shinbaram/data/yolo_module"
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

# ---- 타일/크롭 설정 ----
TILE = 250
OVERLAP = 0.20
PAD_RATIO = 0.05
PAD_MIN = 25
# 타일에 결함 채택 기준 (둘 중 하나라도 만족)
MIN_DEFECT_FRAC = 0.30   # 결함 bbox 넓이 대비 타일 안에 들어온 비율 (작은 porosity 용)
MIN_TILE_FRAC = 0.02     # 타일 넓이 대비 결함 교집합 비율 (큰 resin overflow 용)
SEED = 42

# 형태별 클래스 (id 통일: porosity=0, resin overflow=1)
FORM_CLASSES = {
    "cell_pouch": {"porosity": 0},
    "module_pouch": {"porosity": 0, "resin overflow": 1},
}

NAME_PAT = re.compile(r"CT_(.+?)_(\d+)_[xyz]_\d+")
RUN_PAT = re.compile(r"CT_(.+?)_(\d+)_([xyz])_(\d+)")        # form, num, axis, slice
DEFECTS_NULL = re.compile(r'"defects"\s*:\s*null')
RESIN_RE = re.compile(r'"name"\s*:\s*"resin overflow"')


def battery_key(file_name):
    """CT_module_pouch_006_z_077 -> 'module_pouch_006' (배터리 단위, 분할 키)"""
    m = NAME_PAT.match(file_name)
    return f"{m.group(1)}_{m.group(2)}" if m else file_name


def grp_slice(stem):
    """CT_module_pouch_006_z_077 -> (('module_pouch_006','z'), 77)"""
    m = RUN_PAT.match(stem)
    return (f"{m.group(1)}_{m.group(2)}", m.group(3)), int(m.group(4))


def build_defect_keep(pairs, keep_k):
    """결함 슬라이스를 (배터리,축)별 연속 run 으로 묶고, run 당 균등 keep_k 장만 선택.
    run 이 keep_k 보다 짧으면 전부 보존 -> 짧은 결함도 누락 0.
    ★ resin overflow 슬라이스는 소수 클래스(전체 1868)라 defect-keep 면제(전량 보존).
    반환: (keep 집합 {(grp,slice)}, run수, resin 보유 배터리 set)."""
    by_grp = defaultdict(list)
    keep = set()
    resin_bats = set()
    for img, jp in pairs:
        txt = jp.read_text(encoding="utf-8")
        if DEFECTS_NULL.search(txt):  # 정상 슬라이스
            continue
        grp, sl = grp_slice(img.stem)
        if RESIN_RE.search(txt):      # resin 슬라이스: 전량 보존 + run 축소 제외
            keep.add((grp, sl))
            resin_bats.add(grp[0])
            continue
        by_grp[grp].append(sl)        # porosity 만 run 축소 대상
    n_runs = 0
    for grp, slices in by_grp.items():
        s = sorted(set(slices))
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[j] + 1:
                j += 1
            run = s[i:j + 1]
            n_runs += 1
            L = len(run)
            if L <= keep_k:
                sel = run
            else:
                sel = [run[round(t * (L - 1) / (keep_k - 1))] for t in range(keep_k)]
            for sl in sel:
                keep.add((grp, sl))
            i = j + 1
    return keep, n_runs, resin_bats


def poly_to_bbox(points):
    xs, ys = points[0::2], points[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def pad_box_minwh(box, min_wh):
    """결함 bbox의 폭/높이가 min_wh 미만이면 중심 기준으로 확장.
    porosity 가 폭 3px 가는 박스라 YOLO 가 못 잡는 문제 대응(3px -> min_wh)."""
    x0, y0, x1, y1 = box
    if x1 - x0 < min_wh:
        cx = (x0 + x1) / 2
        x0, x1 = cx - min_wh / 2, cx + min_wh / 2
    if y1 - y0 < min_wh:
        cy = (y0 + y1) / 2
        y0, y1 = cy - min_wh / 2, cy + min_wh / 2
    return [x0, y0, x1, y1]


def pad_crop_box(box, W, H):
    x0, y0, x1, y1 = box
    pad = max(PAD_MIN, int(min(x1 - x0, y1 - y0) * PAD_RATIO))
    return [max(0, int(x0 - pad)), max(0, int(y0 - pad)),
            min(W, int(x1 + pad)), min(H, int(y1 + pad))]


def tile_starts(length, tile, overlap):
    step = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, max(1, length - tile + 1), step))
    if not starts or starts[-1] != length - tile:
        starts.append(max(0, length - tile))
    return sorted(set(starts))


def find_pairs(src, form):
    """src 재귀 탐색 -> 지정 form 의 (image, json) 쌍. .zip 등은 자동 무시(.jpg/.json만)."""
    src = Path(src)
    jsons = {p.stem: p for p in src.rglob("*.json")}
    pairs = []
    prefix = f"CT_{form}_"
    for img in src.rglob("*.jpg"):
        if img.name.startswith(prefix) and img.stem in jsons:
            pairs.append((img, jsons[img.stem]))
    return pairs


def build_slice_index(pairs):
    """(배터리,축) -> {슬라이스번호: img_path}. 2.5D 인접 슬라이스 조회용."""
    idx = defaultdict(dict)
    for img, _ in pairs:
        m = RUN_PAT.match(img.stem)
        if m:
            grp = (f"{m.group(1)}_{m.group(2)}", m.group(3))  # (배터리, 축)
            idx[grp][int(m.group(4))] = img
    return idx


def load_depth_crop(img_path, crop_box, slice_index, use_2p5d):
    """2.5D: 인접 슬라이스(i-1,i,i+1)를 같은 crop_box 로 잘라 R/G/B 채널로 합성.
    인접이 없으면 중앙 슬라이스로 대체(경계). 2D 면 그냥 회색->RGB."""
    cx0, cy0, cx1, cy1 = crop_box
    center = Image.open(img_path).convert("L").crop((cx0, cy0, cx1, cy1))
    if not use_2p5d:
        return Image.merge("RGB", (center, center, center))
    m = RUN_PAT.match(img_path.stem)
    grp = (f"{m.group(1)}_{m.group(2)}", m.group(3))
    sl = int(m.group(4))
    neigh = slice_index.get(grp, {})

    def band(n):
        p = neigh.get(n)
        if p is None:
            return center
        return Image.open(p).convert("L").crop((cx0, cy0, cx1, cy1))

    return Image.merge("RGB", (band(sl - 1), center, band(sl + 1)))


def slice_role(j):
    """슬라이스 역할 분류: 'pos'(결함박스) / 'neg'(완전정상) / 'skip'(스웰링전용·애매)."""
    defs = j.get("defects") or []
    if defs:
        return "pos"
    sw = (j.get("swelling") or {}).get("swelling")
    if sw is True:
        return "skip"  # 스웰링 전용 -> 별도 트랙, detection 에서 제외
    if j.get("image_info", {}).get("is_normal") is True:
        return "neg"
    return "skip"


def take_tile(bx, tx, ty):
    """결함 bbox(bx, 크롭좌표)와 타일(tx,ty,TILE) 교집합 -> 채택시 (cls용 좌표) 반환, 아니면 None."""
    bx0, by0, bx1, by1 = bx
    ix0, iy0 = max(bx0, tx), max(by0, ty)
    ix1, iy1 = min(bx1, tx + TILE), min(by1, ty + TILE)
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return None
    inter = iw * ih
    barea = max(1.0, (bx1 - bx0) * (by1 - by0))
    if inter / barea < MIN_DEFECT_FRAC and inter / (TILE * TILE) < MIN_TILE_FRAC:
        return None
    ncx = ((ix0 + ix1) / 2 - tx) / TILE
    ncy = ((iy0 + iy1) / 2 - ty) / TILE
    return ncx, ncy, iw / TILE, ih / TILE


def process(args):
    random.seed(SEED)
    if args.form not in FORM_CLASSES:
        raise SystemExit(f"--form 은 {list(FORM_CLASSES)} 중 하나여야 함")
    class_map = FORM_CLASSES[args.form]

    out = Path(args.out)
    viz_dir = out / "_viz"
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    if args.viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.src, args.form)
    print(f"[{args.form}] 이미지/라벨 쌍: {len(pairs)}개")
    if not pairs:
        raise SystemExit("쌍을 못 찾음. --src/--form 확인.")

    # 1차: 결함 run 식별 -> run 당 defect_keep 장만 보존(짧은 run 은 전부 -> 불량 누락 0)
    #   resin 슬라이스는 면제(전량 보존), resin 배터리 명단도 반환(stratified split용)
    keep_defect, n_runs, resin_bats = build_defect_keep(pairs, args.defect_keep)
    print(f"결함 run {n_runs}개 -> 보존 결함슬라이스 {len(keep_defect)}개 (run당 최대 {args.defect_keep}장)")
    if resin_bats:
        print(f"resin 보유 배터리 {len(resin_bats)}개(전량보존): {sorted(resin_bats)}")

    # 2.5D: 인접 슬라이스 조회 인덱스 (전체 슬라이스 기준, defect-keep 와 무관)
    slice_index = build_slice_index(pairs) if args.depth_2p5d else {}
    if args.depth_2p5d:
        print(f"2.5D 모드: 인접 슬라이스(i-1,i,i+1) R/G/B 채널 합성. (배터리,축) {len(slice_index)}개")

    # 배터리 단위 train/val 분할 (★ resin stratified: resin 배터리를 train/val 강제 배분)
    by_bat = defaultdict(list)
    for img, jp in pairs:
        by_bat[battery_key(img.name)].append((img, jp))
    bat_ids = sorted(by_bat)
    if resin_bats:
        # resin 배터리: 슬라이스 많은 순(대량 407~411 우선)으로 정렬해 val 에 대량 1개 보장
        resin_ids = sorted((b for b in bat_ids if b in resin_bats),
                           key=lambda b: -len(by_bat[b]))
        other_ids = [b for b in bat_ids if b not in resin_bats]
        random.shuffle(other_ids)
        nv_r = max(1, int(round(len(resin_ids) * args.val_ratio)))
        nv_o = max(1, int(len(other_ids) * args.val_ratio))
        # resin val 은 [대량 1개 + 소량 (nv_r-1)개]로 (대량=리스트 앞)
        val_resin = resin_ids[:1] + resin_ids[-(nv_r - 1):] if nv_r > 1 else resin_ids[:1]
        val_ids = set(val_resin + other_ids[:nv_o])
        print(f"배터리 {len(bat_ids)}개 -> val {len(val_ids)}개 (resin val: {sorted(set(val_resin))})")
    else:
        random.shuffle(bat_ids)
        n_val = max(1, int(len(bat_ids) * args.val_ratio)) if len(bat_ids) > 1 else 0
        val_ids = set(bat_ids[:n_val])
        print(f"배터리 {len(bat_ids)}개 -> val {len(val_ids)}개")

    stats = defaultdict(int)
    cls_count = defaultdict(int)
    skipped_cls = defaultdict(int)
    viz_left = args.viz_n
    done = 0

    for bid in bat_ids:
        split = "val" if bid in val_ids else "train"
        for img_path, jp in by_bat[bid]:
            j = json.loads(jp.read_text(encoding="utf-8"))
            role = slice_role(j)
            stats[f"role_{role}"] += 1
            if role == "skip":
                continue
            # 결함 슬라이스: run 당 defect_keep 장만 (중복 9장 -> 3장, 불량 누락 0)
            if role == "pos":
                grp, sl = grp_slice(img_path.stem)
                if (grp, sl) not in keep_defect:
                    stats["pos_slice_dropped"] += 1
                    continue
            # negative 슬라이스 축소 샘플링
            if role == "neg" and random.random() > args.normal_sample_ratio:
                stats["neg_slice_dropped"] += 1
                continue

            W, H = j["image_info"]["width"], j["image_info"]["height"]
            outline = (j.get("swelling") or {}).get("battery_outline")
            if not outline:
                stats["no_outline"] += 1
                continue
            cx0, cy0, cx1, cy1 = pad_crop_box(poly_to_bbox(outline), W, H)

            # 결함 bbox (크롭 좌표계, 이 form 의 클래스만)
            defects = []
            for d in (j.get("defects") or []):
                nm = d["name"]
                if nm not in class_map:
                    skipped_cls[nm] += 1
                    continue
                bx = poly_to_bbox(d["points"])
                if args.min_box_px > 0:
                    bx = pad_box_minwh(bx, args.min_box_px)
                defects.append((class_map[nm], [bx[0]-cx0, bx[1]-cy0, bx[2]-cx0, bx[3]-cy0]))

            crop = load_depth_crop(img_path, (cx0, cy0, cx1, cy1), slice_index, args.depth_2p5d)
            if args.mask_edge_px > 0:
                import numpy as _np
                _a = _np.array(crop)
                _a[:, :args.mask_edge_px] = 0
                _a[:, -args.mask_edge_px:] = 0
                crop = Image.fromarray(_a)
            cw, ch = crop.size
            xs = tile_starts(cw, TILE, OVERLAP)
            ys = tile_starts(ch, TILE, OVERLAP)
            stem = img_path.stem

            pos_tiles, neg_tiles = [], []
            for ty in ys:
                for tx in xs:
                    labels = []
                    for cls, bx in defects:
                        r = take_tile(bx, tx, ty)
                        if r:
                            labels.append((cls, *r))
                    (pos_tiles if labels else neg_tiles).append((tx, ty, labels))

            # 네거티브 타일 제한: 결함 슬라이스는 pos당 NEG, 정상 슬라이스는 bg_per_normal
            random.shuffle(neg_tiles)
            if role == "pos":
                keep_neg = neg_tiles[: int(len(pos_tiles) * args.neg_ratio)]
            else:
                keep_neg = neg_tiles[: args.bg_per_normal]
            for tx, ty, labels in pos_tiles + keep_neg:
                tile_img = crop.crop((tx, ty, tx + TILE, ty + TILE))
                tname = f"{stem}_t{tx}_{ty}"
                tile_img.save(out / "images" / split / f"{tname}.jpg", quality=90)
                (out / "labels" / split / f"{tname}.txt").write_text(
                    "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels),
                    encoding="utf-8")
                stats[f"{split}_tiles"] += 1
                if labels:
                    stats[f"{split}_pos"] += 1
                    for c, *_ in labels:
                        cls_count[c] += 1
                else:
                    stats[f"{split}_neg"] += 1

            if args.viz and defects and viz_left > 0:
                vv = crop.copy()
                dd = ImageDraw.Draw(vv)
                for _, b in defects:
                    dd.rectangle(b, outline=(255, 0, 0), width=4)
                vv.save(viz_dir / f"{stem}_crop.jpg", quality=85)
                viz_left -= 1

            done += 1
            if done % 5000 == 0:
                print(f"  ...{done} 슬라이스 처리 (tiles train={stats['train_tiles']} val={stats['val_tiles']})")

    # data.yaml
    names = [k for k, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    (out / "battery_ct.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {names}\n", encoding="utf-8")

    print("\n=== 통계 ===")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"  클래스별 인스턴스(id): {dict(cls_count)}  names={names}")
    if skipped_cls:
        print(f"  [제외된 타 클래스] {dict(skipped_cls)}")
    print(f"  data.yaml -> {out/'battery_ct.yaml'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="원본 루트(재귀 탐색; 1.데이터 또는 Sample)")
    ap.add_argument("--form", required=True, choices=list(FORM_CLASSES), help="처리할 형태")
    ap.add_argument("--out", required=True, help="YOLO 출력 폴더")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="val 배터리 비율")
    ap.add_argument("--normal-sample-ratio", type=float, default=0.3,
                    help="완전 정상 슬라이스 사용 비율(결함 중심 축소)")
    ap.add_argument("--defect-keep", type=int, default=3,
                    help="결함 연속구간(run)당 보존할 슬라이스 수(중복 제거, 짧은 run은 전부 보존)")
    ap.add_argument("--min-box-px", type=float, default=0,
                    help="결함 bbox 최소 폭/높이(px). 가는 porosity(3px) 확장용. 0=비활성, 11 권장")
    ap.add_argument("--depth-2p5d", action="store_true",
                    help="2.5D: 인접 슬라이스(i-1,i,i+1)를 R/G/B 채널로. 결함(깊이 9슬라이스)과 정상 배터리구조(깊이 불변) 변별용")
    ap.add_argument("--mask-edge-px", type=int, default=0,
                    help="크롭 좌우 N px 마스킹(배터리 경계 발화 FP 제거용). cell 250폭에 ~38 권장")
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="결함 슬라이스 내 pos당 배경타일 수")
    ap.add_argument("--bg-per-normal", type=int, default=2, help="정상 슬라이스당 배경타일 수")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--viz-n", type=int, default=10)
    process(ap.parse_args())


if __name__ == "__main__":
    main()
