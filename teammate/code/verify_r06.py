"""r06 last.pt 정합성 재검증.
핵심 질문: FP 0.66→0.21 개선이 '진짜 결함을 짚어서'인가, 아니면 '덜 발화해서'인가.
  ① 결함 타일: 예측(빨강)이 GT(초록) 위에 떨어지나 (적중)
  ② 정상 타일: 발화율 + 발화 위치 (배터리영역 전체 발화 해소됐나)
r05_640 대비: 정상 타일에서 배터리영역 통짜 발화가 사라지고, 결함에만 맞으면 성공."""
import glob
import os
import random

from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = "/dev/shm/yolo_cell"
OUT = "/workspace/backup/vr06"
CONF = 0.1
os.makedirs(OUT, exist_ok=True)
m = YOLO("runs/detect/models/cell_r06/weights/last.pt")
val = glob.glob(f"{ROOT}/images/val/*.jpg")

defect_tiles, clean_tiles = [], []
for p in val:
    lp = p.replace("/images/", "/labels/")[:-4] + ".txt"
    if os.path.exists(lp) and os.path.getsize(lp) > 0:
        defect_tiles.append(p)
    else:
        clean_tiles.append(p)
random.Random(1).shuffle(defect_tiles)
random.Random(2).shuffle(clean_tiles)
print(f"결함 타일 {len(defect_tiles)}, 정상 타일 {len(clean_tiles)}")

# ── 정상 타일 발화율 (FP 0.21 정성 확인): 샘플 800장 ──
sample = clean_tiles[:800]
fired = 0
fired_widths = []
for p in sample:
    r = m.predict(p, conf=CONF, imgsz=640, verbose=False)[0]
    if len(r.boxes) > 0:
        fired += 1
        for b in r.boxes.xywhn.tolist():
            fired_widths.append(b[2])
print(f"정상타일 발화율(샘플{len(sample)}): {fired}/{len(sample)} = {fired/len(sample):.3f}")
if fired_widths:
    fw = sorted(fired_widths)
    print(f"  정상 발화박스 폭: 중앙값 {fw[len(fw)//2]:.3f}, 최대 {fw[-1]:.3f} (큰값=배터리영역 통짜발화 의심)")

# ── 결함 타일 적중률: 예측 박스가 GT와 겹치나 ──
def iou_any(pred_boxes, gt_boxes):
    """예측 박스 중 하나라도 GT와 겹치면 True (중심 거리 간이 IoU)."""
    for pb in pred_boxes:
        for gb in gt_boxes:
            # 겹침: 중심 x,y 가 서로 박스 안
            if abs(pb[0]-gb[0]) < (pb[2]+gb[2])/2 and abs(pb[1]-gb[1]) < (pb[3]+gb[3])/2:
                return True
    return False

hit = 0
checked = defect_tiles[:300]
for p in checked:
    lp = p.replace("/images/", "/labels/")[:-4] + ".txt"
    gt = [list(map(float, l.split()))[1:] for l in open(lp) if l.strip()]
    r = m.predict(p, conf=CONF, imgsz=640, verbose=False)[0]
    pb = r.boxes.xywhn.tolist()
    if pb and iou_any(pb, gt):
        hit += 1
print(f"결함타일 적중률(예측이 GT와 겹침, 샘플{len(checked)}): {hit}/{len(checked)} = {hit/len(checked):.3f}")


def draw(p, tag, idx):
    im = Image.open(p).convert("RGB"); W, H = im.size
    dr = ImageDraw.Draw(im)
    lp = p.replace("/images/", "/labels/")[:-4] + ".txt"
    ng = 0
    if os.path.exists(lp):
        for line in open(lp):
            if not line.strip():
                continue
            _, cx, cy, w, h = map(float, line.split())
            dr.rectangle([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H], outline=(0, 255, 0), width=2)
            ng += 1
    r = m.predict(p, conf=CONF, imgsz=640, verbose=False)[0]
    npr = len(r.boxes)
    pws = []
    for b in r.boxes.xywhn.tolist():
        cx, cy, w, h = b
        pws.append(round(w, 3))
        dr.rectangle([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H], outline=(255, 0, 0), width=1)
    im.resize((W*4, H*4), Image.NEAREST).save(f"{OUT}/{tag}_{idx}_gt{ng}_pred{npr}.png")


# 시각화: 결함 6장 + 발화하는 정상 6장
for i, p in enumerate(defect_tiles[:6]):
    draw(p, "defect", i)
shown = 0
for p in clean_tiles:
    r = m.predict(p, conf=CONF, imgsz=640, verbose=False)[0]
    if len(r.boxes) > 0:
        draw(p, "cleanFP", shown)
        shown += 1
        if shown >= 6:
            break
print(f"시각화 저장: {OUT} (결함6 + 정상발화6)")
print("VERIFY_R06_DONE")
