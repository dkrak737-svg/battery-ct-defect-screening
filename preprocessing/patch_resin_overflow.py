"""One-off patch: add missing 'resin overflow' boxes to labels.jsonl.

Why this exists: the first preprocess_local.py run used
CLASS_TO_ID lookup with key 'resin_overflow' (underscore), but the raw
JSON labels use 'resin overflow' (space). So resin_overflow boxes were
silently dropped (0 count instead of expected ~3.3K).

Rather than re-cropping all 201K images (~1h), we recompute the crop
region from the same JSON + outline (deterministic — identical math as
preprocess_local.py) and convert the resin_overflow polygon into crop
coords. Then we append it to the existing record's `boxes` list.

After this, preprocess_local.py's fix (use config.get_class_id) will
naturally include resin_overflow in any future re-runs.

Usage:
    python preprocessing/patch_resin_overflow.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from preprocessing.bbox_utils import (  # noqa: E402
    polygon_to_bbox, compute_crop_region, shift_bbox, clip_bbox,
)


def _label_dirs():
    """All label directories to scan."""
    return [config.TRAIN_LABEL_DIR, config.VAL_LABEL_DIR]


def patch():
    print(f'Reading {config.LABELS_JSONL} ...')
    records = []
    with open(config.LABELS_JSONL, encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    print(f'  {len(records):,} records loaded')

    # index records by slice file name -> record
    fname_to_record = {Path(r['path']).name: r for r in records}

    # scan all JSONs for resin overflow
    print('Scanning JSON labels for "resin overflow" ...')
    json_files = []
    for ldir in _label_dirs():
        json_files.extend(ldir.rglob('*.json'))
    print(f'  total JSON files: {len(json_files):,}')

    n_patched = 0
    n_boxes_added = 0
    n_missing_record = 0
    cls_id_resin = config.CLASS_TO_ID['resin_overflow']
    cls_id_swelling = config.CLASS_TO_ID['swelling']
    pad = config.CROP_PADDING_PX
    min_bbox = config.MIN_BBOX_SIZE_PX
    total = len(json_files)
    t0 = time.time()

    for i, jp in enumerate(json_files):
        if i % 5000 == 0:
            dt = time.time() - t0
            rate = i / dt if dt > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(f'  [{i:>7,} / {total:,}]  rate={rate:5.0f}/s  '
                  f'eta={eta/60:4.1f}min  patched={n_patched:,}  boxes={n_boxes_added:,}',
                  flush=True)
        with open(jp, encoding='utf-8') as f:
            d = json.load(f)
        defects = d.get('defects') or []
        # any resin overflow?
        resin_defects = [df for df in defects
                         if (df.get('name') or '').strip() == 'resin overflow']
        if not resin_defects:
            continue

        slice_fname = jp.stem + '.jpg'
        rec = fname_to_record.get(slice_fname)
        if rec is None:
            n_missing_record += 1
            continue

        # recompute crop region (same logic as preprocess_local.py)
        ii = d.get('image_info', {}) or {}
        sw = d.get('swelling', {}) or {}
        img_w = ii.get('width'); img_h = ii.get('height')
        if not img_w or not img_h:
            continue

        outline_pts = sw.get('battery_outline') or []
        outline_bbox = polygon_to_bbox(outline_pts, min_size=min_bbox) if outline_pts else None

        # recompute the EXACT same union used originally (porosity + resin_overflow + swelling-if-true)
        all_bboxes = []
        for df in defects:
            cls_id = config.get_class_id(df.get('name'))
            if cls_id is None:
                continue
            pts = df.get('points') or []
            bb = polygon_to_bbox(pts, min_size=min_bbox)
            if bb is None: continue
            all_bboxes.append(bb)
        if sw.get('swelling') and outline_bbox is not None:
            all_bboxes.append(outline_bbox)

        crop_x0, crop_y0, crop_x1, crop_y1 = compute_crop_region(
            outline_bbox, all_bboxes, padding=pad,
            img_w=int(img_w), img_h=int(img_h),
        )
        crop_w_new = crop_x1 - crop_x0
        crop_h_new = crop_y1 - crop_y0

        # sanity: must match what's already stored (±1px tolerance)
        if abs(crop_w_new - rec['crop_w']) > 1 or abs(crop_h_new - rec['crop_h']) > 1:
            print(f'  WARN crop mismatch for {slice_fname}: '
                  f'recomputed {crop_w_new}x{crop_h_new} vs stored '
                  f'{rec["crop_w"]}x{rec["crop_h"]}. Skipping.')
            continue

        # convert resin_overflow polygons -> crop-coord bboxes; append to record.
        added_here = 0
        for df in resin_defects:
            pts = df.get('points') or []
            bb = polygon_to_bbox(pts, min_size=min_bbox)
            if bb is None: continue
            shifted = shift_bbox(bb, crop_x0, crop_y0)
            clipped = clip_bbox(shifted, crop_w_new, crop_h_new, min_size=min_bbox)
            if clipped is None: continue
            x0, y0, x1, y1 = clipped
            rec['boxes'].append({
                'cls': cls_id_resin,
                'x0': round(x0, 2), 'y0': round(y0, 2),
                'x1': round(x1, 2), 'y1': round(y1, 2),
            })
            added_here += 1

        if added_here:
            n_patched += 1
            n_boxes_added += added_here

    print(f'\nPatched records: {n_patched:,}')
    print(f'Resin overflow boxes added: {n_boxes_added:,}')
    if n_missing_record:
        print(f'WARN: JSONs whose slice record was not found: {n_missing_record}')

    # write back
    print(f'\nRewriting {config.LABELS_JSONL} ...')
    backup = config.LABELS_JSONL.with_suffix('.jsonl.bak')
    config.LABELS_JSONL.rename(backup)
    with open(config.LABELS_JSONL, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'  done. backup at {backup.name}')


if __name__ == '__main__':
    patch()
