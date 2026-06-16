"""
배터리/슬라이스 단위 OR 집계 평가 (진짜 KPI).

타일 단위 mAP는 작은 객체 IoU에 민감해 무의미하게 출렁인다. 운영 목적(기내 화재 방지)은
"이 배터리를 불량으로 플래그했는가"이므로, 타일 예측을 슬라이스→배터리로 OR 집계해 평가한다.
  - 슬라이스 recall: 결함 슬라이스 중 (타일 하나라도 발화)로 잡은 비율
  - 슬라이스 단위 FP: 정상 슬라이스인데 발화 (오탐)
  - 배터리 recall/precision: 배터리 단위 OR (발화 타일 N개 이상 -> 불량 플래그)

사용:
  python vision/eval_battery.py --weights <best.pt> --data /dev/shm/yolo_cell/battery_ct.yaml \
      --imgsz 256 --confs 0.05,0.1,0.25 --batt-min-tiles 1
"""
import argparse
import os
import re
from collections import defaultdict

import yaml
from ultralytics import YOLO

PAT = re.compile(r"(CT_.+?_\d+)_([xyz]_\d+)_t\d+_\d+$")  # 배터리, 축_슬라이스


def parse_stem(stem):
    m = PAT.match(stem)
    return (m.group(1), m.group(2)) if m else (stem, "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--confs", default="0.05,0.1,0.25")
    ap.add_argument("--batt-min-tiles", type=int, default=1,
                    help="(구버전) 배터리 플래그 최소 발화 타일 수")
    ap.add_argument("--consec-ks", default="1,2,3,5",
                    help="깊이 연속성 k 여러 개 비교(predict 1회로). 인접 k슬라이스 연속 발화 시 불량 플래그")
    ap.add_argument("--max-box-w", type=float, default=1.0,
                    help="이 폭(정규화) 미만 박스만 결함으로 인정. 큰 박스(배터리영역 발화) 필터용. 1.0=필터없음")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.data, encoding="utf-8"))
    root = cfg["path"]
    val_spec = cfg["val"]

    # val 이 폴더(images/val) 또는 txt(이미지경로 리스트) 둘 다 지원
    if val_spec.endswith(".txt"):
        import glob as _g
        val_list = [l.strip() for l in open(os.path.join(root, val_spec)) if l.strip()]
    else:
        import glob as _g
        val_list = _g.glob(os.path.join(root, val_spec, "*.jpg"))
    val_src = val_list  # predict 에 리스트 그대로 전달

    # GT: 타일별 결함 여부(라벨 파일 크기>0)
    gt = {}
    for ip in val_list:
        lp = ip.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        stem = os.path.basename(ip).rsplit(".", 1)[0]
        gt[stem] = os.path.exists(lp) and os.path.getsize(lp) > 0
    n_pos = sum(gt.values())
    print(f"val 타일 {len(gt)} (결함 {n_pos} / 정상 {len(gt)-n_pos})")

    model = YOLO(args.weights)
    print(f"\n{'conf':>5} | {'슬라이스recall':>12} | {'슬라이스FP율':>11} | {'배터리recall':>11} | {'배터리precision':>13}")
    print("-" * 70)

    for conf in [float(x) for x in args.confs.split(",")]:
        import torch
        fired = {}  # tile -> bool (예측 발화)
        # OOM 방지: 작은 청크로 추론. ★ predict(리스트)는 r.path를 'image0.jpg'로 바꾸므로
        #   입력 경로(chunk)와 결과를 zip 으로 매핑(r.path 쓰면 안 됨).
        CHUNK = 32
        for s in range(0, len(val_src), CHUNK):
            chunk = val_src[s:s + CHUNK]
            results = model.predict(chunk, conf=conf, imgsz=args.imgsz, verbose=False)
            for ip, r in zip(chunk, results):
                stem = os.path.basename(ip).rsplit(".", 1)[0]
                # 박스 크기 필터: w < max_box_w 인 박스만 결함으로(큰 박스=배터리영역 무시)
                boxes = r.boxes.xywhn.tolist() if len(r.boxes) else []
                small = [b for b in boxes if b[2] < args.max_box_w]
                fired[stem] = len(small) > 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 슬라이스/배터리 집계
        sl_gt, sl_fire = defaultdict(bool), defaultdict(int)
        bat_gt, bat_fire = defaultdict(bool), defaultdict(int)
        bat_all = set()
        for tile, ispos in gt.items():
            bat, sl = parse_stem(tile)
            key = (bat, sl)
            bat_all.add(bat)
            if ispos:
                sl_gt[key] = True
                bat_gt[bat] = True
            if fired.get(tile):
                sl_fire[key] += 1
                bat_fire[bat] += 1

        # 슬라이스 단위
        pos_sl = [k for k, v in sl_gt.items() if v]
        neg_sl = [k for k in set(list(sl_fire) + [(b, s) for (b, s) in sl_gt]) if not sl_gt[k]]
        sl_recall = sum(1 for k in pos_sl if sl_fire[k] > 0) / max(1, len(pos_sl))
        # 정상 슬라이스 FP율: 전체 슬라이스 중 정상인데 발화
        all_sl = set()
        for tile in gt:
            all_sl.add(parse_stem(tile))
        neg_all = [k for k in all_sl if not sl_gt[k]]
        sl_fp = sum(1 for k in neg_all if sl_fire[k] > 0) / max(1, len(neg_all))

        pos_bat = [b for b, v in bat_gt.items() if v]
        neg_bat = [b for b in bat_all if not bat_gt[b]]

        # ★ 깊이 연속성 규칙: 배터리를 "어떤 축에서 인접 k슬라이스 연속 발화" 시 불량 플래그.
        #   predict 1회 결과로 여러 k 를 한 번에 비교(recall<->FP 트레이드오프).
        fire_ba = defaultdict(lambda: defaultdict(set))  # bat -> axis -> {slice_no}
        for (b, sl), cnt in sl_fire.items():
            if cnt > 0:
                ax, no = sl.rsplit("_", 1)
                fire_ba[b][ax].add(int(no))

        def max_run(b):
            best = 0
            for ss in fire_ba[b].values():
                s = sorted(ss)
                run = 1 if s else 0
                best = max(best, run)
                for i in range(1, len(s)):
                    run = run + 1 if s[i] == s[i - 1] + 1 else 1
                    best = max(best, run)
            return best

        pos_run = {b: max_run(b) for b in pos_bat}
        neg_run = {b: max_run(b) for b in neg_bat}
        ks = [int(x) for x in args.consec_ks.split(",")]
        for k in ks:
            tp = sum(1 for b in pos_bat if pos_run[b] >= k)
            fp = sum(1 for b in neg_bat if neg_run[b] >= k)
            b_recall = tp / max(1, len(pos_bat))
            b_prec = tp / max(1, tp + fp)
            print(f"{conf:>5} | k={k} | slice_R {sl_recall:.3f} | slice_FP {sl_fp:.3f}"
                  f" | bat_R {b_recall:.3f} ({tp}/{len(pos_bat)}) | bat_P {b_prec:.3f}"
                  f" | 정상오플래그 {fp}/{len(neg_bat)}")


if __name__ == "__main__":
    main()
