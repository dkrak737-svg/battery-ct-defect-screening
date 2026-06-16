"""Pure functions: polygon -> bbox + crop region computation.

All functions take/return plain tuples — easy to unit test, no I/O.
"""
from typing import List, Optional, Sequence, Tuple

Bbox = Tuple[float, float, float, float]  # (x0, y0, x1, y1) in pixels


def polygon_to_bbox(
    points: Sequence[float], min_size: float = 2.0
) -> Optional[Bbox]:
    """Flat [x0,y0,x1,y1,...] polygon -> external bbox.
    Returns None if polygon is degenerate or resulting bbox is < min_size on either side.
    """
    if not points or len(points) < 4:
        return None
    xs = points[0::2]
    ys = points[1::2]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    if (x1 - x0) < min_size or (y1 - y0) < min_size:
        return None
    return (float(x0), float(y0), float(x1), float(y1))


def union_bboxes(bboxes: List[Bbox]) -> Optional[Bbox]:
    """Smallest enclosing bbox over a list."""
    if not bboxes:
        return None
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def compute_crop_region(
    outline_bbox: Optional[Bbox],
    defect_bboxes: List[Bbox],
    padding: float,
    img_w: int,
    img_h: int,
) -> Bbox:
    """Crop region = (outline_bbox UNION all defect_bboxes) + padding,
    clipped to image bounds. Returns integer pixel bbox.

    If both outline and defects are missing, falls back to whole image
    (shouldn't happen with our data — stats showed 0 missing outlines).
    """
    boxes: List[Bbox] = list(defect_bboxes)
    if outline_bbox is not None:
        boxes.append(outline_bbox)
    u = union_bboxes(boxes)
    if u is None:
        return (0, 0, img_w, img_h)
    x0, y0, x1, y1 = u
    x0 = max(0, int(x0 - padding))
    y0 = max(0, int(y0 - padding))
    x1 = min(img_w, int(x1 + padding))
    y1 = min(img_h, int(y1 + padding))
    return (x0, y0, x1, y1)


def shift_bbox(bbox: Bbox, dx: float, dy: float) -> Bbox:
    """Translate bbox by (-dx, -dy) — used to convert original coords to crop coords."""
    return (bbox[0] - dx, bbox[1] - dy, bbox[2] - dx, bbox[3] - dy)


def clip_bbox(
    bbox: Bbox, w: float, h: float, min_size: float = 2.0
) -> Optional[Bbox]:
    """Clip bbox to (0,0,w,h). Return None if result is too small or zero-area."""
    x0 = max(0.0, bbox[0])
    y0 = max(0.0, bbox[1])
    x1 = min(float(w), bbox[2])
    y1 = min(float(h), bbox[3])
    if (x1 - x0) < min_size or (y1 - y0) < min_size:
        return None
    return (x0, y0, x1, y1)


# ---------- quick self-test (run as: python preprocessing/bbox_utils.py) ----------
if __name__ == '__main__':
    # polygon_to_bbox
    assert polygon_to_bbox([0, 0, 10, 0, 10, 20, 0, 20]) == (0.0, 0.0, 10.0, 20.0)
    assert polygon_to_bbox([5, 5]) is None  # too few points
    assert polygon_to_bbox([0, 0, 1, 1]) is None  # too small (<2px)

    # union_bboxes
    assert union_bboxes([(0,0,5,5), (3,3,10,10)]) == (0, 0, 10, 10)
    assert union_bboxes([]) is None

    # compute_crop_region
    assert compute_crop_region(
        (100, 200, 300, 600), [(150, 250, 250, 500)], padding=10,
        img_w=1000, img_h=1000
    ) == (90, 190, 310, 610)
    # crop region clipped to image edge
    assert compute_crop_region(
        (0, 0, 990, 990), [], padding=50, img_w=1000, img_h=1000
    ) == (0, 0, 1000, 1000)

    # shift_bbox + clip_bbox
    assert shift_bbox((100, 200, 300, 600), 50, 100) == (50, 100, 250, 500)
    assert clip_bbox((10, 10, 100, 100), 50, 50) == (10, 10, 50, 50)
    assert clip_bbox((-10, -10, 1, 1), 50, 50) is None  # too small after clip

    print('bbox_utils self-tests: OK')
