"""
YOLO11 학습 (배터리 CT 결함 탐지, 형태별 모델)

- recall 최우선: 학습은 표준, 추론/평가 시 conf 낮춰 재현율 확보(val_recall.py 참고).
- 타일 250px -> imgsz 256 기본.
- cell(nc=1 porosity) / module(nc=2 porosity,resin overflow) 공용. --data 로 구분.

사용(RunPod):
  python vision/train.py --data data/yolo_cell/battery_ct.yaml --name cell_r01
  python vision/train.py --data data/yolo_module/battery_ct.yaml --name module_r01 --epochs 80
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="battery_ct.yaml 경로")
    ap.add_argument("--weights", default="yolo11n.pt", help="사전학습 가중치(전이학습)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=256, help="타일 250 -> 256")
    ap.add_argument("--batch", type=int, default=64, help="5090 24GB 기준. OOM 시 낮춤")
    ap.add_argument("--project", default="models")
    ap.add_argument("--name", required=True, help="run 이름(예: cell_r01)")
    ap.add_argument("--patience", type=int, default=20, help="early stop")
    ap.add_argument("--lr0", type=float, default=0.01, help="초기 lr(전이학습 미세조정은 0.003 권장)")
    ap.add_argument("--warmup-epochs", type=float, default=3.0, help="warmup epoch 수")
    ap.add_argument("--optimizer", default="SGD", help="옵티마이저(auto는 lr0 무시하므로 SGD 고정)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from ultralytics import YOLO

    if not Path(args.data).exists():
        raise SystemExit(f"data yaml 없음: {args.data}")

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        seed=42,
        # 전이학습 미세조정: optimizer 고정(auto는 lr0 무시함!) + lr 낮춤 + warmup 짧게
        optimizer=args.optimizer,
        lr0=args.lr0,
        warmup_epochs=args.warmup_epochs,
        # 작은/가는 결함(porosity) 위해 모자이크 후반 비활성 + 약한 증강
        close_mosaic=10,
        # 검증 지표에 recall 포함(기본 출력). 저장: best.pt = 최고 mAP 기준이지만
        # 실제 운용 conf 는 추론 단계에서 낮춰 recall 확보.
        plots=True,
    )
    print(f"\n학습 완료. 가중치: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
