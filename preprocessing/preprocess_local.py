"""Local preprocessing: original CT slices -> cropped JPGs + labels.jsonl.

Runs LOCALLY (Windows). Uses multiprocessing for speed (~3h for 201K slices on 8 cores).
After this completes, tar up cropped/ by battery and upload to RunPod.

Usage:
    python preprocessing/preprocess_local.py --limit 100   # smoke test
    python preprocessing/preprocess_local.py               # full run
"""
import argparse
import json
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# project root on sys.path so `config` + `preprocessing.bbox_utils` import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from preprocessing.bbox_utils import (  # noqa: E402
    polygon_to_bbox, compute_crop_region, shift_bbox, clip_bbox,
)


# ---------- Unicode-safe cv2 I/O (Windows Korean paths) ----------
def imread_u(path: Path):
    """cv2.imread that handles Unicode paths on Windows."""
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imwrite_u(path: Path, img, params=None) -> bool:
    """cv2.imwrite that handles Unicode paths on Windows."""
    ext = path.suffix
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


# ---------- Job collection ----------
def collect_jobs() -> List[Tuple[str, Path, Path]]:
    """Find all (set, json_path, image_path) triplets.
    Returns list of jobs ready for the worker pool.
    """
    jobs: List[Tuple[str, Path, Path]] = []

    # Build image_name -> path index (faster than per-json glob)
    print('Indexing image files...')
    train_idx = {p.name: p for p in config.TRAIN_IMG_DIR.rglob('*.jpg')}
    val_idx   = {p.name: p for p in config.VAL_IMG_DIR.rglob('*.jpg')}
    print(f'  TRAIN images: {len(train_idx):,}')
    print(f'  VAL   images: {len(val_idx):,}')

    print('Indexing label files...')
    train_labels = list(config.TRAIN_LABEL_DIR.rglob('*.json'))
    val_labels   = list(config.VAL_LABEL_DIR.rglob('*.json'))
    print(f'  TRAIN labels: {len(train_labels):,}')
    print(f'  VAL   labels: {len(val_labels):,}')

    missing = 0
    for jp in train_labels:
        ip = train_idx.get(jp.stem + '.jpg')
        if ip is None:
            missing += 1
            continue
        jobs.append(('TRAIN', jp, ip))
    for jp in val_labels:
        ip = val_idx.get(jp.stem + '.jpg')
        if ip is None:
            missing += 1
            continue
        jobs.append(('VAL', jp, ip))

    print(f'  Pairs found: {len(jobs):,}  (missing image: {missing:,})')
    return jobs


# ---------- Worker ----------
def _battery_key(set_name: str, btype: str, bform: str, bid) -> str:
    """Deterministic, OS-safe folder name per battery."""
    if isinstance(bid, int):
        return f'{set_name}_{btype}_{bform}_{bid:04d}'
    return f'{set_name}_{btype}_{bform}_{bid}'


def process_one(args):
    """Worker: process a single (json, image) pair. Picklable for mp.Pool.
    Returns dict (success record) or {'error': ...} or None (skipped).
    """
    set_name, json_path, image_path, output_root, padding, min_bbox, jpg_quality = args
    try:
        with open(json_path, encoding='utf-8') as f:
            d = json.load(f)

        di = d.get('data_info', {})  or {}
        ii = d.get('image_info', {}) or {}
        sw = d.get('swelling', {})   or {}
        defects = d.get('defects') or []

        img_w = ii.get('width')
        img_h = ii.get('height')
        if not img_w or not img_h:
            return None

        # outline_bbox from battery_outline polygon
        outline_pts = sw.get('battery_outline') or []
        outline_bbox = polygon_to_bbox(outline_pts, min_size=min_bbox) if outline_pts else None

        # defect bboxes (porosity / resin_overflow)
        records: List[Tuple[int, tuple]] = []
        for df in defects:
            cls_id = config.get_class_id(df.get('name'))
            if cls_id is None:
                continue
            pts = df.get('points') or []
            bbox = polygon_to_bbox(pts, min_size=min_bbox)
            if bbox is None:
                continue
            records.append((cls_id, bbox))

        # swelling: per-slice flag; bbox = battery_outline
        if sw.get('swelling') and outline_bbox is not None:
            cls_id = config.CLASS_TO_ID.get('swelling')
            if cls_id is not None:
                records.append((cls_id, outline_bbox))

        # crop region = union(outline + defects) + padding
        all_bboxes = [b for _, b in records]
        crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_region(
            outline_bbox, all_bboxes,
            padding=padding, img_w=int(img_w), img_h=int(img_h),
        )
        crop_w = crop_x1 - crop_x0
        crop_h = crop_y1 - crop_y0
        if crop_w <= 0 or crop_h <= 0:
            return None

        # read image (Unicode-safe), crop
        img = imread_u(image_path)
        if img is None:
            return {'error': 'imread failed', 'json': str(json_path)}
        cropped = img[crop_y0:crop_y1, crop_x0:crop_x1]

        # output path: cropped/<battery_key>/<basename>.jpg
        btype = di.get('type', 'unknown')
        bform = di.get('form', 'unknown')
        bid   = di.get('battery_ids', 'X')
        bkey  = _battery_key(set_name, btype, bform, bid)

        out_dir = Path(output_root) / bkey
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / image_path.name

        ok = imwrite_u(out_path, cropped,
                       [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        if not ok:
            return {'error': 'imwrite failed', 'json': str(json_path)}

        # boxes in CROP coordinates
        boxes_out = []
        for cls_id, bbox in records:
            shifted = shift_bbox(bbox, crop_x0, crop_y0)
            clipped = clip_bbox(shifted, crop_w, crop_h, min_size=min_bbox)
            if clipped is None:
                continue
            x0, y0, x1, y1 = clipped
            boxes_out.append({
                'cls': cls_id,
                'x0': round(x0, 2), 'y0': round(y0, 2),
                'x1': round(x1, 2), 'y1': round(y1, 2),
            })

        return {
            'set'           : set_name,
            'path'          : f'{bkey}/{image_path.name}',
            'battery_key'   : bkey,
            'battery_id'    : bid,
            'type'          : btype,
            'form'          : bform,
            'is_normal'     : bool(ii.get('is_normal')),
            'swelling_flag' : bool(sw.get('swelling')),
            'crop_w'        : int(crop_w),
            'crop_h'        : int(crop_h),
            'boxes'         : boxes_out,
        }
    except Exception as e:
        return {'error': repr(e), 'json': str(json_path),
                'trace': traceback.format_exc()[-500:]}


def main(limit: Optional[int] = None):
    output_root = config.CROPPED_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    print(f'Output dir: {output_root}')
    print(f'Labels jsonl: {config.LABELS_JSONL}')
    print(f'Padding: {config.CROP_PADDING_PX}px, JPG quality: {config.JPG_QUALITY}')
    print()

    jobs = collect_jobs()
    if limit:
        jobs = jobs[:limit]
        print(f'LIMIT applied: {len(jobs)} jobs')

    args_iter = [
        (set_name, jp, ip, output_root,
         config.CROP_PADDING_PX, config.MIN_BBOX_SIZE_PX, config.JPG_QUALITY)
        for set_name, jp, ip in jobs
    ]

    records, errors = [], []
    print(f'Processing with {config.NUM_WORKERS} workers...')
    with Pool(config.NUM_WORKERS) as pool:
        for rec in tqdm(
            pool.imap_unordered(process_one, args_iter, chunksize=50),
            total=len(args_iter), desc='Cropping', unit='slice',
        ):
            if rec is None:
                continue
            if 'error' in rec:
                errors.append(rec)
                continue
            records.append(rec)

    print(f'\nDone. {len(records):,} OK, {len(errors)} errors')

    # write labels.jsonl
    print(f'Writing {config.LABELS_JSONL} ...')
    with open(config.LABELS_JSONL, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    if errors:
        err_path = config.LABELS_JSONL.with_name('preprocess_errors.jsonl')
        with open(err_path, 'w', encoding='utf-8') as f:
            for e in errors[:200]:  # cap log
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
        print(f'Wrote first {min(200, len(errors))} errors to {err_path}')

    print('OK.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None,
                   help='Process only first N slices (smoke test).')
    args = p.parse_args()
    main(limit=args.limit)
