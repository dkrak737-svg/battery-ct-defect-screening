"""swelling 분류 배터리 단위 평가. 슬라이스 정확도가 아니라 배터리 단위가 진짜 KPI.
  - 슬라이스: val 이미지별 정상/swelling 예측 → 슬라이스 정확도/혼동행렬
  - 배터리: 배터리별 swelling 예측 슬라이스 비율 > 임계값 → swelling 배터리 플래그
    배터리 GT = 그 배터리가 swelling 폴더에 슬라이스를 가지면 swelling 배터리.
"""
import argparse
import glob
import os
import re
from collections import defaultdict

from ultralytics import YOLO

BAT = re.compile(r"CT_module_pouch_(\d+)_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="yolo_swelling/images 폴더")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batt-thr", default="0.1,0.3,0.5", help="배터리 플래그 swelling 비율 임계값들")
    args = ap.parse_args()

    val = args.data.rstrip("/") + "/val"
    imgs, gt = [], {}  # path -> 1(swelling)/0(normal)
    for cls, lab in (("normal", 0), ("swelling", 1)):
        for p in glob.glob(f"{val}/{cls}/*.jpg"):
            imgs.append(p); gt[p] = lab
    print(f"val 슬라이스 {len(imgs)} (swelling {sum(gt.values())} / normal {len(imgs)-sum(gt.values())})")

    model = YOLO(args.weights)
    pred = {}
    CH = 256
    import torch
    for s in range(0, len(imgs), CH):
        chunk = imgs[s:s + CH]
        for p, r in zip(chunk, model.predict(chunk, imgsz=args.imgsz, verbose=False)):
            pred[p] = int(r.probs.top1)  # 0/1 (names 순: normal,swelling 가정 -> top1 인덱스)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # names 인덱스 확인 (normal=0, swelling=1 보장)
    names = model.names
    sw_idx = [k for k, v in names.items() if v == "swelling"][0]

    # 슬라이스 혼동행렬
    tp = sum(1 for p in imgs if gt[p] == 1 and pred[p] == sw_idx)
    fn = sum(1 for p in imgs if gt[p] == 1 and pred[p] != sw_idx)
    fp = sum(1 for p in imgs if gt[p] == 0 and pred[p] == sw_idx)
    tn = sum(1 for p in imgs if gt[p] == 0 and pred[p] != sw_idx)
    print(f"\n[슬라이스] swelling recall {tp/max(1,tp+fn):.3f} | normal recall {tn/max(1,tn+fp):.3f} "
          f"| 정확도 {(tp+tn)/len(imgs):.3f}  (TP{tp} FN{fn} FP{fp} TN{tn})")

    # 배터리 집계
    bat_sw_pred = defaultdict(int); bat_tot = defaultdict(int); bat_gt = defaultdict(int)
    for p in imgs:
        b = BAT.search(os.path.basename(p)).group(1)
        bat_tot[b] += 1
        if pred[p] == sw_idx:
            bat_sw_pred[b] += 1
        if gt[p] == 1:
            bat_gt[b] = 1  # 이 배터리는 swelling 배터리
    pos_b = [b for b in bat_tot if bat_gt[b] == 1]
    neg_b = [b for b in bat_tot if bat_gt[b] == 0]
    print(f"\n[배터리] val swelling 배터리 {len(pos_b)} / 정상 배터리 {len(neg_b)}")
    for thr in [float(x) for x in args.batt_thr.split(",")]:
        btp = sum(1 for b in pos_b if bat_sw_pred[b] / bat_tot[b] > thr)
        bfp = sum(1 for b in neg_b if bat_sw_pred[b] / bat_tot[b] > thr)
        print(f"  thr {thr}: swelling 배터리 recall {btp}/{len(pos_b)}={btp/max(1,len(pos_b)):.3f} "
              f"| 정상 오플래그 {bfp}/{len(neg_b)}")


if __name__ == "__main__":
    main()
