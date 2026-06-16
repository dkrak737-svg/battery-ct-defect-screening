# -*- coding: utf-8 -*-
"""porosity seg 타일링 (RunPod 실행). 멀티프로세싱·스트리밍 저장.
입력: cropped + seg_labels.jsonl(porosity 폴리곤) + labels.jsonl(전체=background) + splits.json
출력: yolo_seg/{train,val,test}/{images,labels}/ + dataset.yaml

설계:
 - 스케일: cell(width~15px) vs module(~85px) 차이는 물리적 모양차로 보고 공통 리사이즈 안 함 → scale 증강.
   (평가에서 cell/module recall 분리 확인 필요)
 - background: porosity 없는 타일을 확률(--bg_keep_prob)로 포함 → 과검출 방지.
 - oversample: porosity 타일 복제(--oversample).
 - 폴리곤은 타일 경계에서 shapely로 정확히 clip → normalized YOLO seg.
"""
import json, os, argparse, random, hashlib
from pathlib import Path
from multiprocessing import Pool
from PIL import Image
from shapely.geometry import Polygon, box
from shapely.validation import make_valid

OPT = {}

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cropped", default="data/cropped")
    ap.add_argument("--seg_labels", default="data/seg_labels.jsonl")
    ap.add_argument("--labels", default="data/labels.jsonl")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--out", default="data/yolo_seg")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--bg_keep_prob", type=float, default=0.15, help="porosity 없는 타일 저장 확률")
    ap.add_argument("--cell_bg_prob", type=float, default=0.15, help="[train split의 cell만] background 저장 확률. cell FP 잡기용(val/test는 bg_keep_prob 고정).")
    ap.add_argument("--oversample", type=int, default=1, help="porosity 타일 복제 수")
    ap.add_argument("--min_area", type=float, default=16.0, help="타일 내 폴리곤 조각 최소 면적(px^2)")
    ap.add_argument("--cell_scale", type=int, default=4, help="cell 등방 확대 배율(얇은 porosity 살림, 모양 보존)")
    ap.add_argument("--module_frac", type=float, default=1.0, help="module 슬라이스 사용 비율(cell은 항상 풀). 속도용.")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="스모크: 앞 N 슬라이스만")
    ap.add_argument("--only_split", default="", help="특정 set만 타일(예: test). eval용 빠른 재생성.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def tile_origins(size, tile, stride):
    if size <= tile:
        return [0]
    xs = list(range(0, size - tile + 1, stride))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return xs

def work(job):
    path, bkey, st, polys = job
    o = OPT
    src = os.path.join(o["cropped"], path)
    try:
        im = Image.open(src).convert("RGB")
    except Exception:
        return (0, 0, 0)
    # cell 등방 확대 (얇은 porosity 살림, 양축 같은 배율이라 모양 보존). module은 scale=1.
    scale = o["cell_scale"] if ("_cell_" in bkey) else 1
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale))
    W, H = im.size
    stride = int(o["tile"] * (1 - o["overlap"]))
    # ⚠️ hash(path)는 PYTHONHASHSEED로 프로세스마다 달라짐(비결정) → md5로 결정화(재타일해도 동일 타일=채점지 고정)
    rng = random.Random(int(hashlib.md5(path.encode()).hexdigest()[:8], 16))

    poly_objs = []
    for pl in polys:
        pts = [(pl[i] * scale, pl[i + 1] * scale) for i in range(0, len(pl) - 1, 2)]
        if len(pts) >= 3:
            g = Polygon(pts)
            if not g.is_valid:
                g = make_valid(g)
            if g.geom_type == "Polygon" and g.area > 0:
                poly_objs.append(g)

    n_pos = n_bg = n_miss = 0
    for oy in tile_origins(H, o["tile"], stride):
        for ox in tile_origins(W, o["tile"], stride):
            tbox = box(ox, oy, ox + o["tile"], oy + o["tile"])
            lines = []
            for g in poly_objs:
                if not g.intersects(tbox):
                    continue
                inter = g.intersection(tbox)
                if inter.is_empty:
                    continue
                geoms = [inter] if inter.geom_type == "Polygon" else list(getattr(inter, "geoms", []))
                for gg in geoms:
                    if gg.geom_type != "Polygon" or gg.area < o["min_area"]:
                        continue
                    coords = list(gg.exterior.coords)
                    norm = []
                    for (x, y) in coords:
                        norm.append(round((x - ox) / o["tile"], 6))
                        norm.append(round((y - oy) / o["tile"], 6))
                    lines.append("0 " + " ".join(str(v) for v in norm))
            is_pos = bool(lines)
            if not is_pos:
                # train split의 cell만 cell_bg_prob 적용(FP 잡기). val/test·module은 bg_keep_prob 고정.
                prob = o["cell_bg_prob"] if (st == "train" and "_cell_" in bkey) else o["bg_keep_prob"]
                if rng.random() >= prob:
                    continue
            # 타일 추출 (가장자리 pad)
            crop = im.crop((ox, oy, ox + o["tile"], oy + o["tile"]))
            if crop.size != (o["tile"], o["tile"]):
                cv = Image.new("RGB", (o["tile"], o["tile"]), (0, 0, 0))
                cv.paste(crop, (0, 0))
                crop = cv
            base = path.replace("/", "__").rsplit(".", 1)[0] + f"_{ox}_{oy}"
            dups = o["oversample"] if is_pos else 1
            for d in range(dups):
                stem = base + (f"_d{d}" if d else "")
                ip = os.path.join(o["out"], st, "images", stem + ".jpg")
                lp = os.path.join(o["out"], st, "labels", stem + ".txt")
                crop.save(ip, quality=90)
                with open(lp, "w") as f:
                    f.write("\n".join(lines))
            if is_pos:
                n_pos += 1
            else:
                n_bg += 1
    return (n_pos, n_bg, n_miss)

def main():
    a = parse()
    OPT.update(dict(cropped=a.cropped, out=a.out, tile=a.tile, overlap=a.overlap,
                    bg_keep_prob=a.bg_keep_prob, cell_bg_prob=a.cell_bg_prob,
                    oversample=a.oversample, min_area=a.min_area,
                    cell_scale=a.cell_scale))
    split = json.load(open(a.splits, encoding="utf-8"))
    seg = {}
    for line in open(a.seg_labels, encoding="utf-8"):
        r = json.loads(line)
        seg[r["path"]] = r["polygons"]

    def set_of(bkey):
        return split.get(bkey.split("_", 1)[1])

    import random as _rnd
    mrng = _rnd.Random(a.seed)
    jobs = []
    for line in open(a.labels, encoding="utf-8"):
        r = json.loads(line.strip())
        st = set_of(r["battery_key"])
        if st is None:
            continue
        if a.only_split and st != a.only_split:
            continue
        # module만 subsample (cell은 minority+약자라 항상 풀)
        if "_module_" in r["battery_key"] and mrng.random() >= a.module_frac:
            continue
        jobs.append((r["path"], r["battery_key"], st, seg.get(r["path"], [])))
    if a.limit:
        jobs = jobs[:a.limit]

    for s in ["train", "val", "test"]:
        for k in ["images", "labels"]:
            os.makedirs(os.path.join(a.out, s, k), exist_ok=True)

    print(f"슬라이스 {len(jobs)} | tile {a.tile} overlap {a.overlap} bg_prob {a.bg_keep_prob} oversample {a.oversample}")
    tp = tb = 0
    with Pool(a.workers) as pool:
        for i, (np_, nb_, _) in enumerate(pool.imap_unordered(work, jobs, chunksize=20), 1):
            tp += np_; tb += nb_
            if i % 10000 == 0:
                print(f"  {i}/{len(jobs)}  pos타일 {tp:,}  bg타일 {tb:,}")
    print(f"완료. pos타일 {tp:,} (oversample x{a.oversample}) + bg타일 {tb:,}")

    with open(os.path.join(a.out, "dataset.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\n")
        f.write("train: train/images\nval: val/images\ntest: test/images\n")
        f.write("names:\n  0: porosity\n")
    # 폴더별 수
    for s in ["train", "val", "test"]:
        d = os.path.join(a.out, s, "images")
        print(f"  {s}: {len(os.listdir(d)) if os.path.isdir(d) else 0} tiles")

if __name__ == "__main__":
    main()
