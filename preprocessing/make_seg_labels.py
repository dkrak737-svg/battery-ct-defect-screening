# -*- coding: utf-8 -*-
"""전체 zip(Training+Validation) -> porosity seg 라벨 (crop 좌표 polygon).
이미지 불필요 (좌표 계산만). preprocess_local.py와 동일 crop 로직 -> 기존 cropped와 정렬.
출력: seg_labels.jsonl  (slice별: path, crop_w/h, polygons[ porosity crop좌표 flat ])
"""
import sys, json, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.bbox_utils import polygon_to_bbox, compute_crop_region

BASE = r"C:\Users\User\Desktop\103.배터리 불량 이미지 데이터\3.개방데이터\1.데이터"
ZIPS = {
    "TRAIN": BASE + r"\Training\02.라벨링데이터\TL_CT_Datasets_label.zip",
    "VAL":   BASE + r"\Validation\02.라벨링데이터\VL_CT_Datasets_label.zip",
}
OUT = Path(r"C:\Users\User\Desktop\Sati\CT_DATA_handoff\seg_labels.jsonl")
PAD = 50      # config.CROP_PADDING_PX
MIN = 2.0     # config.MIN_BBOX_SIZE_PX

def battery_key(set_name, btype, bform, bid):
    if isinstance(bid, int):
        return f"{set_name}_{btype}_{bform}_{bid:04d}"
    return f"{set_name}_{btype}_{bform}_{bid}"

def clip_poly(pts, cx0, cy0, w, h):
    out = []
    for i in range(0, len(pts) - 1, 2):
        x = min(max(pts[i] - cx0, 0.0), w)
        y = min(max(pts[i + 1] - cy0, 0.0), h)
        out.extend([round(x, 2), round(y, 2)])
    return out

n_slice = n_poly = 0
batt = set()
with open(OUT, "w", encoding="utf-8") as fo:
    for set_name, zp in ZIPS.items():
        z = zipfile.ZipFile(zp)
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            d = json.loads(z.read(name))
            defects = d.get("defects") or []
            poros = [df for df in defects if df.get("name") == "porosity" and df.get("points")]
            if not poros:
                continue  # porosity 없는 슬라이스는 seg 대상 아님
            di = d.get("data_info", {}) or {}
            ii = d.get("image_info", {}) or {}
            sw = d.get("swelling", {}) or {}
            img_w, img_h = int(ii["width"]), int(ii["height"])
            outline = sw.get("battery_outline") or []
            outline_bbox = polygon_to_bbox(outline, MIN) if outline else None
            defect_bboxes = []
            for df in defects:
                bb = polygon_to_bbox(df.get("points") or [], MIN)
                if bb:
                    defect_bboxes.append(bb)
            all_bb = list(defect_bboxes)
            if sw.get("swelling") and outline_bbox is not None:
                all_bb.append(outline_bbox)
            cx0, cy0, cx1, cy1 = compute_crop_region(outline_bbox, all_bb, PAD, img_w, img_h)
            crop_w, crop_h = cx1 - cx0, cy1 - cy0
            if crop_w <= 0 or crop_h <= 0:
                continue
            bkey = battery_key(set_name, di.get("type", "unknown"), di.get("form", "unknown"), di.get("battery_ids", "X"))
            polys = []
            for df in poros:
                cp = clip_poly(df["points"], cx0, cy0, crop_w, crop_h)
                if len(cp) >= 6:  # 최소 3점
                    polys.append(cp)
            if not polys:
                continue
            img_name = ii.get("file_name") or Path(name).name.replace(".json", ".jpg")
            fo.write(json.dumps({
                "set": set_name,
                "path": f"{bkey}/{img_name}",
                "battery_key": bkey,
                "crop_w": crop_w, "crop_h": crop_h,
                "polygons": polys,
            }, ensure_ascii=False) + "\n")
            n_slice += 1
            n_poly += len(polys)
            batt.add(bkey.split("_", 1)[1])

print("porosity 슬라이스:", n_slice)
print("porosity 폴리곤:", n_poly)
print("porosity 배터리:", len(batt))
print("->", OUT)
