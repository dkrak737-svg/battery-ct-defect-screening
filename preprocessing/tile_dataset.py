"""Tile cropped slices into standard YOLO dataset folder.

Runs on RunPod (after cropped/ + labels.jsonl are uploaded).
Reads:  config.CROPPED_DIR + config.LABELS_JSONL
Writes: <DATA_ROOT>/yolo_data/
        ├── dataset.yaml
        ├── images/{train,val,test}/<bkey>__<slice>__t<r>_<c>.jpg
        └── labels/{train,val,test}/<bkey>__<slice>__t<r>_<c>.txt

Re-runnable: changing tile params and re-running rebuilds from cropped/.

Usage:
    python preprocessing/tile_dataset.py
    python preprocessing/tile_dataset.py --limit 100   # smoke test
"""
import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# ---------- Unicode-safe cv2 I/O ----------
def imread_u(path: Path):
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imwrite_u(path: Path, img, params=None) -> bool:
    ok, buf = cv2.imencode(path.suffix, img, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


# ---------- Tile geometry ----------
def tile_positions(dim: int, tile: int, stride: int) -> List[int]:
    """Positions for tile top-left along one axis.
    Always covers the entire dim; last tile is anchored to (dim - tile) so we don't miss the edge.
    If dim <= tile, returns [0] (single tile, image will be padded).
    """
    if dim <= tile:
        return [0]
    pos = list(range(0, dim - tile + 1, stride))
    if pos[-1] != dim - tile:
        pos.append(dim - tile)
    return pos


def pad_to_tile(img: np.ndarray, tile: int) -> np.ndarray:
    """Pad image with black (0) on bottom/right so dims >= tile."""
    h, w = img.shape[:2]
    pad_h = max(0, tile - h)
    pad_w = max(0, tile - w)
    if pad_h == 0 and pad_w == 0:
        return img
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)


def clip_and_normalize_box(
    box: Dict, tx0: int, ty0: int, tile: int, min_size: float
) -> Optional[Tuple[int, float, float, float, float]]:
    """Box is in crop coords. Convert to tile coords, clip to tile, normalize to [0,1].
    Returns (cls, cx, cy, w, h) or None if box doesn't intersect tile (or too small).
    """
    x0, y0, x1, y1 = box['x0'] - tx0, box['y0'] - ty0, box['x1'] - tx0, box['y1'] - ty0
    # No intersection with tile?
    if x1 <= 0 or y1 <= 0 or x0 >= tile or y0 >= tile:
        return None
    # Clip
    x0 = max(0.0, x0); y0 = max(0.0, y0)
    x1 = min(float(tile), x1); y1 = min(float(tile), y1)
    w = x1 - x0; h = y1 - y0
    if w < min_size or h < min_size:
        return None
    # Normalize to [0,1]
    cx = (x0 + x1) / 2 / tile
    cy = (y0 + y1) / 2 / tile
    return (box['cls'], cx, cy, w / tile, h / tile)


# ---------- Split ----------
def split_batteries(records: List[Dict], ratios: Dict[str, float], seed: int) -> Dict[str, str]:
    """Stratified split by (type, form), grouping by *physical* battery
    so TRAIN_xxx and VAL_xxx for the same physical battery go to the same split.

    Returns {battery_key: split_name} (covers BOTH TRAIN_/VAL_ variants).
    """
    # physical key = (type, form, battery_id) — same regardless of TRAIN/VAL origin
    phys_to_bkeys: Dict[Tuple, set] = defaultdict(set)
    group_to_phys: Dict[Tuple, set] = defaultdict(set)
    for r in records:
        phys = (r['type'], r['form'], r['battery_id'])
        phys_to_bkeys[phys].add(r['battery_key'])
        group_to_phys[(r['type'], r['form'])].add(phys)

    rng = random.Random(seed)
    phys_to_split: Dict[Tuple, str] = {}
    print('\nSplit per (type, form) - by physical battery:')
    for group, phys_set in group_to_phys.items():
        phys_list = sorted(phys_set)
        rng.shuffle(phys_list)
        n = len(phys_list)
        n_train = int(n * ratios['train'])
        n_val   = int(n * ratios['val'])
        for p in phys_list[:n_train]:
            phys_to_split[p] = 'train'
        for p in phys_list[n_train:n_train + n_val]:
            phys_to_split[p] = 'val'
        for p in phys_list[n_train + n_val:]:
            phys_to_split[p] = 'test'
        print(f'  {group}: total={n}  train={n_train}  val={n_val}  test={n - n_train - n_val}')

    # Map every battery_key in the dataset to its physical key's split
    bkey_to_split: Dict[str, str] = {}
    for phys, bkeys in phys_to_bkeys.items():
        split = phys_to_split[phys]
        for bk in bkeys:
            bkey_to_split[bk] = split
    return bkey_to_split


# ---------- Worker ----------
def process_one(args):
    """Worker: tile one slice. Returns (n_tiles_written, error_msg_or_None)."""
    (rec, split_name, cropped_root, out_images, out_labels,
     tile, stride, min_box, jpg_quality, resin_oversample) = args
    try:
        img_path = Path(cropped_root) / rec['path']
        img = imread_u(img_path)
        if img is None:
            return (0, f'imread fail: {img_path}')

        h, w = img.shape[:2]
        # Sanity: cropped dim should match record (allow small JPG roundtrip diff)
        if rec.get('crop_w') and abs(rec['crop_w'] - w) > 2:
            return (0, f'crop_w mismatch: rec={rec["crop_w"]} got={w}')
        if rec.get('crop_h') and abs(rec['crop_h'] - h) > 2:
            return (0, f'crop_h mismatch: rec={rec["crop_h"]} got={h}')

        ys = tile_positions(h, tile, stride)
        xs = tile_positions(w, tile, stride)

        # Pad if smaller than tile
        if h < tile or w < tile:
            img = pad_to_tile(img, tile)

        boxes = rec.get('boxes') or []
        slice_stem = Path(rec['path']).stem  # e.g., CT_cell_pouch_101_y_033
        bkey = rec['battery_key']

        n_written = 0
        for r, ty0 in enumerate(ys):
            for c, tx0 in enumerate(xs):
                tile_img = img[ty0:ty0 + tile, tx0:tx0 + tile]
                # safety: pad if (rare) still under-size due to padding edge
                if tile_img.shape[0] != tile or tile_img.shape[1] != tile:
                    tile_img = pad_to_tile(tile_img, tile)

                # collect normalized boxes for this tile
                yolo_lines = []
                for b in boxes:
                    out = clip_and_normalize_box(b, tx0, ty0, tile, min_box)
                    if out is None:
                        continue
                    cls, cx, cy, bw, bh = out
                    yolo_lines.append(f'{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

                # r02: oversample rare resin_overflow class
                resin_id = config.CLASS_TO_ID['resin_overflow']
                has_resin = any(int(line.split(' ')[0]) == resin_id for line in yolo_lines)
                n_copies = resin_oversample if (has_resin and split_name == 'train') else 1

                base_name = f'{bkey}__{slice_stem}__t{r}_{c}'
                for copy_idx in range(n_copies):
                    suffix = f'_dup{copy_idx}' if copy_idx > 0 else ''
                    name = base_name + suffix
                    img_path_out = Path(out_images) / split_name / f'{name}.jpg'
                    lbl_path_out = Path(out_labels) / split_name / f'{name}.txt'

                    if not imwrite_u(img_path_out, tile_img, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]):
                        return (n_written, f'imwrite fail: {img_path_out}')
                    # Empty label file is valid YOLO (background-only tile)
                    with open(lbl_path_out, 'w', encoding='utf-8') as f:
                        if yolo_lines:
                            f.write('\n'.join(yolo_lines) + '\n')
                    n_written += 1

        return (n_written, None)
    except Exception as e:
        import traceback
        return (0, f'{e!r}\n{traceback.format_exc()[-500:]}')


# ---------- Main ----------
def main(limit: Optional[int] = None, clean: bool = False):
    labels_path = config.LABELS_JSONL
    if not labels_path.exists():
        print(f'ERROR: {labels_path} not found. Run preprocess_local.py first.')
        sys.exit(1)

    out_root  = config.DATA_ROOT / 'yolo_data'
    out_images = out_root / 'images'
    out_labels = out_root / 'labels'

    if clean and out_root.exists():
        print(f'Cleaning {out_root} ...')
        shutil.rmtree(out_root)
    for split in ('train', 'val', 'test'):
        (out_images / split).mkdir(parents=True, exist_ok=True)
        (out_labels / split).mkdir(parents=True, exist_ok=True)

    print(f'Reading {labels_path} ...')
    records = []
    with open(labels_path, encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    print(f'  {len(records):,} slice records')

    if limit:
        records = records[:limit]
        print(f'LIMIT applied: {len(records)} records')

    bkey_to_split = split_batteries(records, config.SPLIT_RATIOS, config.RANDOM_SEED)

    # Build worker arg list
    args_iter = []
    for rec in records:
        split = bkey_to_split.get(rec['battery_key'])
        if split is None:
            continue
        args_iter.append((
            rec, split, config.CROPPED_DIR, out_images, out_labels,
            config.TILE_SIZE, config.TILE_STRIDE, config.MIN_TILE_BBOX_PX,
            config.JPG_QUALITY, config.RESIN_OVERFLOW_OVERSAMPLE,
        ))

    print(f'\nTiling with {config.NUM_WORKERS} workers '
          f'(tile={config.TILE_SIZE}, stride={config.TILE_STRIDE}) ...')
    n_tiles_total = 0
    errors: List[str] = []
    with Pool(config.NUM_WORKERS) as pool:
        for n, err in tqdm(
            pool.imap_unordered(process_one, args_iter, chunksize=50),
            total=len(args_iter), desc='Tiling', unit='slice',
        ):
            n_tiles_total += n
            if err:
                errors.append(err)

    print(f'\nDone. {n_tiles_total:,} tiles written, {len(errors)} errors.')

    # dataset.yaml for Ultralytics
    yaml_path = out_root / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"# Auto-generated by tile_dataset.py\n")
        f.write(f"path: {out_root.as_posix()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/val\n")
        f.write(f"test:  images/test\n")
        f.write(f"nc: {config.NUM_CLASSES}\n")
        f.write(f"names: {config.CLASSES}\n")
    print(f'Wrote {yaml_path}')

    if errors:
        err_path = out_root / 'tile_errors.log'
        with open(err_path, 'w', encoding='utf-8') as f:
            for e in errors[:200]:
                f.write(e + '\n')
        print(f'First {min(200, len(errors))} errors -> {err_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None,
                   help='Process only first N slice records')
    p.add_argument('--clean', action='store_true',
                   help='Delete existing yolo_data/ before tiling')
    args = p.parse_args()
    main(limit=args.limit, clean=args.clean)
