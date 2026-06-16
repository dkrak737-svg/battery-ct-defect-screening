# -*- coding: utf-8 -*-
"""porosity seg 배터리 단위 평가 (운영 지표).
1순위: 배터리 단위 porosity recall — cell/module 분리.
+ conf×k 2D 스윕: 외딴 오탐 타일(k로 제거 가능) vs 진짜 못 본 놓침(k로 못 살림) 구분.
+ 배터리별 검출 타일 수 덤프 → 놓친 module / 오탐 음성 정체 파악.
※ ultralytics mask mAP는 'yolo segment val'로 따로 — 모니터링용일 뿐.
"""
import os, glob, argparse
from collections import defaultdict
from ultralytics import YOLO

THRS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
KS = [1, 2, 3, 5, 10]

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/seg/porosity/weights/best.pt")
    ap.add_argument("--data", default="data/yolo_seg")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1024)
    return ap.parse_args()

def battery_of(p):
    bkey = os.path.basename(p).split("__")[0]   # TRAIN_cell_pouch_0101
    return bkey.split("_", 1)[1]                 # cell_pouch_0101

def main():
    a = parse()
    img_dir = os.path.join(a.data, a.split, "images")
    lbl_dir = os.path.join(a.data, a.split, "labels")

    # GT: 배터리별 porosity 보유 여부 (타일 라벨 non-empty 하나라도)
    gt = {}
    for lp in glob.glob(os.path.join(lbl_dir, "*.txt")):
        core = battery_of(lp)
        gt[core] = gt.get(core, False) or (os.path.getsize(lp) > 0)

    # 추론: 타일별 검출 conf → 배터리별 thr별 '검출 타일 수'
    model = YOLO(a.weights)
    ndet = defaultdict(lambda: defaultdict(int))   # core -> thr -> 검출타일수
    ntile = defaultdict(int)
    # conf=0.05 floor: THRS 최저가 0.05라 결과 불변 + 과검출 모델의 저신뢰 박스 폭발(NMS 부하) 방지.
    for r in model.predict(img_dir, imgsz=a.imgsz, conf=0.05, stream=True, verbose=False):
        core = battery_of(r.path)
        ntile[core] += 1
        confs = [float(c) for c in r.boxes.conf] if (r.boxes is not None) else []
        for thr in THRS:
            if any(c >= thr for c in confs):
                ndet[core][thr] += 1

    cores = sorted(gt)
    is_cell = lambda c: c.startswith("cell")
    npos = sum(1 for c in cores if gt[c])
    nneg = sum(1 for c in cores if not gt[c])
    ncellpos = sum(1 for c in cores if gt[c] and is_cell(c))
    nmodpos = npos - ncellpos
    print(f"[{a.split}] 배터리 {len(cores)} (양성 {npos}: cell {ncellpos}/mod {nmodpos} / 음성 {nneg})")

    # ── conf × k 2D 스윕 ──
    for k in KS:
        print(f"\n=== k={k} (검출 타일 >= {k} 면 배터리 flag) ===")
        print("conf  | rec(all) rec(cell) rec(mod) | fpr   fp/neg")
        for thr in THRS:
            tp = fn = fp = tn = 0
            cc = [0, 0]; cm = [0, 0]
            for c in cores:
                pred = ndet[c].get(thr, 0) >= k
                if gt[c]:
                    if is_cell(c): cc[1] += 1; cc[0] += int(pred)
                    else: cm[1] += 1; cm[0] += int(pred)
                    tp += int(pred); fn += int(not pred)
                else:
                    fp += int(pred); tn += int(not pred)
            rec = tp / (tp + fn) if tp + fn else 0.0
            rcc = cc[0] / cc[1] if cc[1] else 0.0
            rcm = cm[0] / cm[1] if cm[1] else 0.0
            fpr = fp / (fp + tn) if fp + tn else 0.0
            print("%.2f  |  %.3f    %.3f     %.3f   |  %.3f  %d/%d"
                  % (thr, rec, rcc, rcm, fpr, fp, fp + tn))

    # ── 배터리별 검출 타일 수 (정체 파악) ──
    print("\n=== 배터리별 검출 타일 수 ===")
    print("battery                | type GT  | ntile | det@.05 .10 .15 .25")
    for c in cores:
        typ = "cell" if is_cell(c) else "mod "
        g = "pos" if gt[c] else "neg"
        print("%-22s | %s %s | %5d | %5d %4d %4d %4d"
              % (c, typ, g, ntile[c],
                 ndet[c].get(0.05, 0), ndet[c].get(0.10, 0),
                 ndet[c].get(0.15, 0), ndet[c].get(0.25, 0)))
    print("\n해석:")
    print("  GT=pos & det@.05=0  → 모델이 아예 못 본 '놓침'(k로 못 살림 = recall 천장 원인).")
    print("  GT=neg & det>0      → 오탐. 외딴 1~2타일이면 k 상향으로 제거 가능.")
    print("  → 진짜 porosity(세로 띠)는 det 수십, 오탐은 1~2 → 표에서 분리선 보임.")

if __name__ == "__main__":
    main()
