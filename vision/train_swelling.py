"""timm swelling 분류기 — module 전용.

설계 (CLAUDE.md 확정):
 - 입력: 저장된 512 레터박스 그대로. ⚠️ CROP 없음 (RandomResizedCrop/CenterCrop 금지) — 부풀음 곡률은 가장자리에 있음.
 - 불균형(90.5% swell): class-weighted CE, nonswell ~8x.
 - 과적합 방어(실질표본 module 68배터리): pretrained 백본 freeze + head부터 학습(--unfreeze_blocks로 점진 unfreeze) + wd + dropout + label smoothing + early stop.
 - 체크포인트 선택 = val "슬라이스" 지표(수천 장 → 안정). 배터리 지표(음성 3~4개)는 거치니 최종 test 운영 보고로만.
 - 평가: 슬라이스 recall/특이도 + 배터리 집계(≥k장, ⚠️ any 금지) recall/특이도. 목표 = recall≥99% 고정 + 특이도 최대.

사용: python3 vision/train_swelling.py --smoke      # 파이프라인 빠른 검증
      python3 vision/train_swelling.py               # 본학습
"""
import argparse, os, json
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
KS = [1, 3, 5, 10, 20]          # 배터리 ≥k장 임계값 후보

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/swell_cls")
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--nonswell_w", type=float, default=8.0)
    ap.add_argument("--unfreeze_blocks", type=int, default=0)   # 0=head만, N=마지막 N블록도 학습
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/swell")
    return ap.parse_args()

def build_tf(sz):
    # ⚠️ crop 없음. Resize((sz,sz))는 이미 정사각이라 사실상 no-op(안전장치), squish 아님.
    train_tf = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf

def battery_of(path):
    fn = os.path.basename(path)            # TRAIN_module_pouch_0006__CT_..._112.jpg
    return fn.split("__")[0].split("_", 1)[1]   # -> module_pouch_0006

def subset_fraction(ds, frac, seed=42):
    if frac >= 1.0:
        return ds
    g = torch.Generator().manual_seed(seed)
    n = int(len(ds) * frac)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)

@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    probs, labels, idxs = [], [], []
    for x, y, ii in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda"):
            logit = model(x)
        p = torch.softmax(logit.float(), 1)[:, 1]   # P(swell)
        probs.append(p.cpu().numpy()); labels.append(y.numpy()); idxs.append(ii.numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(idxs)

def slice_metrics(probs, labels, thr=0.5):
    pred = (probs >= thr).astype(int)
    pos = labels == 1; neg = labels == 0
    tp = int((pred[pos] == 1).sum()); fn = int((pred[pos] == 0).sum())
    tn = int((pred[neg] == 0).sum()); fp = int((pred[neg] == 1).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(recall=rec, spec=spec, prec=prec, f1=f1)

def spec_at_recall(probs, labels, target=0.99):
    """체크포인트 선택용(슬라이스): recall>=target 만족하며 특이도 최대."""
    pos = probs[labels == 1]; neg = probs[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    thr = np.quantile(pos, 1 - target)        # 이 thr 이상을 swell로 -> recall≈target
    return float((neg < thr).mean())          # 특이도

def battery_eval(probs, labels, paths, ks, thr=0.5):
    pred = (probs >= thr).astype(int)
    by = defaultdict(lambda: dict(true=0, predpos=0))
    for i, p in enumerate(paths):
        b = battery_of(p)
        if labels[i] == 1: by[b]["true"] += 1     # 실제 라벨: any-positive (데이터 사실)
        if pred[i] == 1: by[b]["predpos"] += 1
    res = {}
    for k in ks:
        tp = fp = tn = fn = 0
        for d in by.values():
            actual = d["true"] > 0
            predicted = d["predpos"] >= k         # ⚠️ 예측 집계는 ≥k (any 금지)
            if actual and predicted: tp += 1
            elif actual: fn += 1
            elif predicted: fp += 1
            else: tn += 1
        rec = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        res[k] = (rec, spec, tp, fn, tn, fp)
    return res, len(by)

class IdxImageFolder(datasets.ImageFolder):
    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return x, y, i

def main():
    a = parse()
    if a.smoke:
        a.epochs = 2; a.fraction = min(a.fraction, 0.05)
    os.makedirs(a.out, exist_ok=True)
    device = "cuda"
    train_tf, eval_tf = build_tf(a.imgsz)

    tr = IdxImageFolder(os.path.join(a.data, "train"), transform=train_tf)
    va = IdxImageFolder(os.path.join(a.data, "val"), transform=eval_tf)
    te = IdxImageFolder(os.path.join(a.data, "test"), transform=eval_tf)
    print("classes(알파벳순):", tr.classes)   # ['nonswell','swell'] -> nonswell=0, swell=1
    va_paths = [s[0] for s in va.samples]
    te_paths = [s[0] for s in te.samples]

    tr_ds = subset_fraction(tr, a.fraction)
    va_ds = subset_fraction(va, a.fraction) if a.smoke else va
    dl = lambda d, sh: DataLoader(d, batch_size=a.batch, shuffle=sh, num_workers=a.workers, pin_memory=True, drop_last=sh)
    tr_dl, va_dl, te_dl = dl(tr_ds, True), dl(va_ds, False), dl(te, False)
    # 과적합 격차 확인용: train을 eval transform(augment X)으로 ~5000장 평가
    tr_eval = IdxImageFolder(os.path.join(a.data, "train"), transform=eval_tf)
    tr_eval_dl = dl(subset_fraction(tr_eval, min(1.0, 5000 / len(tr_eval))), False)

    model = timm.create_model(a.backbone, pretrained=True, num_classes=2, drop_rate=a.drop).to(device)
    # freeze 전체 -> head만 학습 (+옵션: 마지막 N블록)
    for p in model.parameters(): p.requires_grad = False
    for p in model.get_classifier().parameters(): p.requires_grad = True
    if a.unfreeze_blocks > 0 and hasattr(model, "blocks"):
        for p in model.blocks[-a.unfreeze_blocks:].parameters(): p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("학습 파라미터: %.2fM / 전체 %.2fM" % (n_train/1e6, sum(p.numel() for p in model.parameters())/1e6))

    w = torch.tensor([a.nonswell_w, 1.0], device=device)   # [nonswell=0, swell=1]
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=a.label_smooth)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_score, best_ep, bad = -1.0, -1, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for x, y, _ in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * x.size(0)
        sched.step()
        # --- val 슬라이스(체크포인트 기준) + train 격차(과적합) + 배터리(참고) ---
        vp, vl, vi = infer(model, va_dl, device)
        vpaths = [va_paths[j] for j in vi]
        sm = slice_metrics(vp, vl)
        score = spec_at_recall(vp, vl, 0.99)         # 체크포인트 선택 = 슬라이스 spec@recall99
        tep, tel, _ = infer(model, tr_eval_dl, device)
        tr_score = spec_at_recall(tep, tel, 0.99)    # train 과적합 격차용
        bev, nb = battery_eval(vp, vl, vpaths, KS)
        b1 = bev[1]; b5 = bev[5] if 5 in bev else bev[KS[-1]]
        print("ep%02d loss%.3f | spec@99 train%.3f val%.3f gap%+.3f | val rec%.3f spec%.3f f1%.3f | batt k1(r%.2f s%.2f) k5(r%.2f s%.2f) n=%d"
              % (ep, tot/max(1,len(tr_ds)), tr_score, score, tr_score-score, sm["recall"], sm["spec"], sm["f1"], b1[0], b1[1], b5[0], b5[1], nb))
        if score > best_score:
            best_score, best_ep, bad = score, ep, 0
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop (patience %d)" % a.patience); break

    # --- 최종 test (best 로드) ---
    print("\n=== BEST ep%d (val spec@rec99=%.3f) -> TEST ===" % (best_ep, best_score))
    if os.path.exists(os.path.join(a.out, "best.pt")):
        model.load_state_dict(torch.load(os.path.join(a.out, "best.pt")))
    tp_, tl_, ti_ = infer(model, te_dl, device)
    tpaths = [te_paths[j] for j in ti_]
    sm = slice_metrics(tp_, tl_)
    print("[test slice] recall %.3f / spec %.3f / f1 %.3f / spec@rec99 %.3f"
          % (sm["recall"], sm["spec"], sm["f1"], spec_at_recall(tp_, tl_, 0.99)))
    bev, nb = battery_eval(tp_, tl_, tpaths, KS)
    print("[test battery] (음성 배터리 적어 거침) n=%d" % nb)
    for k in KS:
        r, s, tp, fn, tn, fp = bev[k]
        print("   k=%-2d  recall %.3f  spec %.3f  (tp%d fn%d tn%d fp%d)" % (k, r, s, tp, fn, tn, fp))

if __name__ == "__main__":
    main()
