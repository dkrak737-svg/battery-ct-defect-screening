"""swelling 이진 분류 학습 (YOLO11-cls). 전역 변화라 detection 아닌 분류로.
  - 입력: data/yolo_swelling/images (train/val/{normal,swelling})
  - 평가 헤드라인은 슬라이스 정확도가 아니라 배터리 단위(eval_swelling.py).
  - 회전 증강은 끔(swelling 외형이 핵심 단서라 왜곡 방지)."""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="yolo_swelling/images 폴더")
    ap.add_argument("--weights", default="yolo11n-cls.pt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--optimizer", default="SGD")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="models",
        name=args.name,
        patience=args.patience,
        device=args.device,
        seed=42,
        optimizer=args.optimizer,
        lr0=args.lr0,
        # swelling 외형 단서 보존: 회전/원근 끔, 좌우반전·약한 밝기만
        degrees=0.0, shear=0.0, perspective=0.0,
        fliplr=0.5, flipud=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        plots=True,
    )
    print(f"\n학습 완료: models/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
