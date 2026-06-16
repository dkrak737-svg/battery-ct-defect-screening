"""Slice-level inference: tile -> YOLO predict -> aggregate -> NMS.

The model trains on 1024 tiles but operational input is a full slice.
This script handles the tile/aggregate dance.

Usage:
    # single image (auto-tile)
    python vision/infer_slice.py --model runs/detect/r01/weights/best.pt \\
        --image path/to/slice.jpg --out out.jpg --conf 0.05

    # batch over a folder
    python vision/infer_slice.py --model best.pt --dir cropped/test/ --out predictions.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from preprocessing.tile_dataset import (  # noqa: E402
    tile_positions, pad_to_tile, imread_u, imwrite_u,
)


def is_mostly_black(tile: np.ndarray, threshold: int = 5, frac: float = 0.99) -> bool:
    """Skip tiles where >=frac of pixels have intensity <=threshold."""
    gray = tile if tile.ndim == 2 else cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    return (gray <= threshold).mean() >= frac


def slice_to_tiles(img: np.ndarray, tile: int, stride: int
                   ) -> List[Tuple[np.ndarray, int, int]]:
    """Yield (tile_image, x0, y0) covering the whole input.
    Pads if image smaller than tile. Skips mostly-black tiles to speed up.
    """
    h, w = img.shape[:2]
    if h < tile or w < tile:
        img = pad_to_tile(img, tile)
        h, w = img.shape[:2]
    out = []
    for ty0 in tile_positions(h, tile, stride):
        for tx0 in tile_positions(w, tile, stride):
            t = img[ty0:ty0+tile, tx0:tx0+tile]
            if t.shape[0] != tile or t.shape[1] != tile:
                t = pad_to_tile(t, tile)
            if is_mostly_black(t):
                continue
            out.append((t, tx0, ty0))
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5
        ) -> np.ndarray:
    """Plain NMS using torchvision. boxes: (N,4) xyxy, scores: (N,)."""
    from torchvision.ops import nms as _nms
    if len(boxes) == 0:
        return np.array([], dtype=int)
    keep = _nms(torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(scores, dtype=torch.float32),
                iou_thresh).numpy()
    return keep


def detect_slice(model, img: np.ndarray, tile: int, stride: int,
                 conf: float, iou: float, device='cpu'):
    """Returns dict with arrays: boxes (N,4) xyxy in slice coords, scores (N,), cls (N,)."""
    tiles = slice_to_tiles(img, tile, stride)
    if not tiles:
        return dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), cls=np.zeros(0, int))

    all_boxes, all_scores, all_cls = [], [], []
    # Batch tiles for one predict call when possible
    tile_imgs = [t for t, _, _ in tiles]
    offsets   = [(x0, y0) for _, x0, y0 in tiles]

    results = model.predict(tile_imgs, conf=conf, iou=iou, device=device,
                            verbose=False, imgsz=tile)
    for (x0, y0), res in zip(offsets, results):
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy   = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        clses  = res.boxes.cls.cpu().numpy().astype(int)
        # shift to slice coords
        xyxy[:, [0, 2]] += x0
        xyxy[:, [1, 3]] += y0
        all_boxes.append(xyxy)
        all_scores.append(scores)
        all_cls.append(clses)

    if not all_boxes:
        return dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), cls=np.zeros(0, int))

    boxes  = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    clses  = np.concatenate(all_cls)

    # Class-aware NMS: dedupe within each class
    keep_all = []
    for c in np.unique(clses):
        m = clses == c
        idx = np.where(m)[0]
        k = nms(boxes[m], scores[m], iou_thresh=iou)
        keep_all.extend(idx[k].tolist())
    keep_all = np.array(sorted(keep_all))
    return dict(boxes=boxes[keep_all], scores=scores[keep_all], cls=clses[keep_all])


def draw_boxes(img: np.ndarray, det: dict, names) -> np.ndarray:
    out = img.copy()
    colors = [(0, 200, 0), (0, 200, 200), (0, 100, 255)]  # porosity/resin/swelling
    for box, s, c in zip(det['boxes'], det['scores'], det['cls']):
        x0, y0, x1, y1 = box.astype(int)
        col = colors[int(c) % len(colors)]
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 3)
        label = f'{names[int(c)]} {s:.2f}'
        cv2.putText(out, label, (x0, max(0, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='YOLO weights .pt')
    p.add_argument('--image', help='single image path')
    p.add_argument('--dir',   help='directory of images (batch mode)')
    p.add_argument('--out',   help='output path: .jpg (single) or .json (batch)')
    p.add_argument('--tile',  type=int, default=config.TILE_SIZE)
    p.add_argument('--stride',type=int, default=config.TILE_STRIDE)
    p.add_argument('--conf',  type=float, default=0.05,
                   help='confidence threshold — LOW for recall')
    p.add_argument('--iou',   type=float, default=0.5)
    p.add_argument('--device',default=0)
    args = p.parse_args()

    if not (args.image or args.dir):
        sys.exit('Need --image or --dir')

    from ultralytics import YOLO
    model = YOLO(args.model)

    if args.image:
        img = imread_u(Path(args.image))
        if img is None:
            sys.exit(f'cannot read {args.image}')
        det = detect_slice(model, img, args.tile, args.stride,
                           args.conf, args.iou, args.device)
        print(f'Detected {len(det["boxes"])} boxes')
        for c in np.unique(det['cls']):
            n = (det['cls'] == c).sum()
            print(f'  {config.CLASSES[int(c)]}: {n}')
        if args.out:
            vis = draw_boxes(img, det, config.CLASSES)
            imwrite_u(Path(args.out), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f'wrote {args.out}')
    else:
        results = []
        for ip in sorted(Path(args.dir).rglob('*.jpg')):
            img = imread_u(ip)
            if img is None: continue
            det = detect_slice(model, img, args.tile, args.stride,
                               args.conf, args.iou, args.device)
            results.append({
                'path' : str(ip),
                'boxes': det['boxes'].tolist(),
                'scores': det['scores'].tolist(),
                'cls'  : det['cls'].tolist(),
            })
        out = args.out or 'predictions.json'
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)
        print(f'wrote {out}  ({len(results)} slices)')


if __name__ == '__main__':
    main()
