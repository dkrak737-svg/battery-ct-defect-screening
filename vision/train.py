"""YOLO11 training wrapper for battery CT defect detection.

Runs on RunPod after tile_dataset.py has produced yolo_data/.
Uses Ultralytics directly — no custom Dataset (on-the-fly tiling was scrapped
because pre-tiled is simpler and tile params still tunable via re-running
tile_dataset.py).

Usage:
    python vision/train.py                     # full run with config defaults
    python vision/train.py --epochs 1 --fraction 0.05  # smoke test
    python vision/train.py --resume runs/.../weights/last.pt   # resume
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',    default=config.YOLO_MODEL_BASE,
                   help='pretrained weights (e.g., yolo11n.pt, yolo11s.pt)')
    p.add_argument('--epochs',   type=int, default=config.EPOCHS)
    p.add_argument('--imgsz',    type=int, default=config.IMG_SZ)
    p.add_argument('--batch',    type=int, default=config.BATCH)
    p.add_argument('--name',     default='r01',
                   help='run name (saved under runs/detect/<name>/)')
    p.add_argument('--fraction', type=float, default=1.0,
                   help='use only fraction of training set (smoke test)')
    p.add_argument('--resume',   default=None,
                   help='resume from a weights .pt')
    p.add_argument('--device',   default=0,
                   help='CUDA device index, "cpu", or comma list')
    p.add_argument('--workers',  type=int, default=8)
    p.add_argument('--lr0',      type=float, default=config.LR0)
    p.add_argument('--warmup_epochs', type=int, default=config.WARMUP_EPOCHS)
    p.add_argument('--amp',      type=lambda x: x.lower() == 'true',
                   default=config.AMP, help='AMP (mixed precision). r02: false')
    p.add_argument('--patience', type=int, default=config.PATIENCE)
    args = p.parse_args()

    from ultralytics import YOLO  # heavy import — keep inside main

    data_yaml = config.DATA_ROOT / 'yolo_data' / 'dataset.yaml'
    if not data_yaml.exists():
        sys.exit(f'ERROR: {data_yaml} not found. Run tile_dataset.py first.')

    model = YOLO(args.resume if args.resume else args.model)

    print(f'\n=== Training ===')
    print(f'  data:     {data_yaml}')
    print(f'  model:    {args.model}')
    print(f'  epochs:   {args.epochs}')
    print(f'  imgsz:    {args.imgsz}')
    print(f'  batch:    {args.batch}')
    print(f'  lr0:      {args.lr0}')
    print(f'  warmup:   {args.warmup_epochs} epochs')
    print(f'  amp:      {args.amp}')
    print(f'  patience: {args.patience}')
    print(f'  fraction: {args.fraction}')
    print(f'  device:   {args.device}')
    print()

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        fraction=args.fraction,
        seed=config.RANDOM_SEED,
        lr0=args.lr0,                       # r02: 0.005 (was 0.01) — prevent ep4 drop
        warmup_epochs=args.warmup_epochs,   # r02: 5 (was 3) — gentler warmup
        amp=args.amp,                       # r02: False — kill box_loss=inf
        patience=args.patience,             # r02: 10 (was 15)
        save_period=5,
    )

    # quick validation summary
    print('\n=== Final Validation ===')
    metrics = model.val(data=str(data_yaml), split='val', imgsz=args.imgsz,
                        device=args.device)
    print(metrics)

    print('\nDone. Best weights:', model.trainer.best)


if __name__ == '__main__':
    main()
