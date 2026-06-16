"""best.pt로 swelling 평가만 (재학습 X).
수정 2가지:
 1) 배터리 집계 버그 수정 — 음성(nonswell) 배터리도 무조건 집계에 포함(by.setdefault).
    (기존 버그: 음성 배터리가 완벽 예측되면 dict 접근이 안 돼 누락 → tn 0).
 2) 음성 모듈 개별 특이도 출력 — "어느 모듈이 어려운지" 평균 한 숫자보다 유용.
데이터는 /workspace 볼륨 직접(추론 1회라 mfs도 OK) → 재복사 불필요.

사용: python3 vision/eval_swelling.py
"""
import argparse, os
import numpy as np, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
KS = [1, 3, 5, 10, 20]

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/battery-ct-security/data/swell_cls")
    ap.add_argument("--weights", default="runs/swell_uf2/best.pt")
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    return ap.parse_args()

class IdxImageFolder(datasets.ImageFolder):
    def __getitem__(self, i):
        x, y = super().__getitem__(i); return x, y, i

def battery_of(path):
    return os.path.basename(path).split("__")[0].split("_", 1)[1]

@torch.no_grad()
def infer(model, loader, device):
    model.eval(); P = []; L = []; I = []
    for x, y, ii in loader:
        x = x.to(device, non_blocking=True)
        logit = model(x)
        P.append(torch.softmax(logit.float(), 1)[:, 1].cpu().numpy()); L.append(y.numpy()); I.append(ii.numpy())
    return np.concatenate(P), np.concatenate(L), np.concatenate(I)

def slice_metrics(probs, labels, thr=0.5):
    pred = (probs >= thr).astype(int); pos = labels == 1; neg = labels == 0
    tp = int((pred[pos] == 1).sum()); fn = int((pred[pos] == 0).sum())
    tn = int((pred[neg] == 0).sum()); fp = int((pred[neg] == 1).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(recall=rec, spec=spec, f1=f1)

def battery_eval(probs, labels, paths, ks, thr=0.5):
    pred = (probs >= thr).astype(int)
    by = {}
    for i, p in enumerate(paths):
        b = battery_of(p)
        d = by.setdefault(b, dict(true=0, predpos=0, n=0, neg=0, fp=0))  # 무조건 생성 (버그 수정)
        d["n"] += 1
        if labels[i] == 1:
            d["true"] += 1
        else:
            d["neg"] += 1
            if pred[i] == 1:
                d["fp"] += 1
        if pred[i] == 1:
            d["predpos"] += 1
    res = {}
    for k in ks:
        tp = fp = tn = fn = 0
        for d in by.values():
            actual = d["true"] > 0; predicted = d["predpos"] >= k
            if actual and predicted: tp += 1
            elif actual: fn += 1
            elif predicted: fp += 1
            else: tn += 1
        rec = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        res[k] = (rec, spec, tp, fn, tn, fp)
    return res, by

def main():
    a = parse(); device = "cuda" if torch.cuda.is_available() else "cpu"; print("device:", device)
    tf = transforms.Compose([transforms.Resize((a.imgsz, a.imgsz)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    model = timm.create_model(a.backbone, pretrained=False, num_classes=2).to(device)
    model.load_state_dict(torch.load(a.weights, map_location=device))
    print("loaded:", a.weights)
    for split in ["val", "test"]:
        ds = IdxImageFolder(os.path.join(a.data, split), transform=tf)
        paths = [s[0] for s in ds.samples]
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True)
        P, L, I = infer(model, dl, device)
        pth = [paths[j] for j in I]
        sm = slice_metrics(P, L)
        print("\n===== %s =====" % split.upper())
        print("[slice] recall %.3f / spec %.3f / f1 %.3f  (음성 슬라이스 %d장)" % (sm["recall"], sm["spec"], sm["f1"], int((L == 0).sum())))
        res, by = battery_eval(P, L, pth, KS)
        negs = {b: d for b, d in by.items() if d["true"] == 0}
        poss = sum(1 for d in by.values() if d["true"] > 0)
        print("[battery] 총 %d개 (양성 %d / 음성 %d)" % (len(by), poss, len(negs)))
        for k in KS:
            r, s, tp, fn, tn, fp = res[k]
            print("   k=%-2d  recall %.3f  spec %.3f   (tp%d fn%d tn%d fp%d)" % (k, r, s, tp, fn, tn, fp))
        print("  -- 음성 모듈 개별 (슬라이스 특이도 = 안 부푼 슬라이스를 맞힌 비율) --")
        for b, d in sorted(negs.items()):
            sl_spec = (d["neg"] - d["fp"]) / d["neg"] if d["neg"] else 0.0
            flag = "  <-- 약함" if sl_spec < 0.9 else ""
            print("    %-22s 슬라이스특이도 %.3f (음성 %d장 중 오탐 %d) | swell예측 %d/%d%s" % (b, sl_spec, d["neg"], d["fp"], d["predpos"], d["n"], flag))

if __name__ == "__main__":
    main()
