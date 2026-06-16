"""module 전용 swelling 분류 데이터셋 구성 (yolo11-cls / timm ImageFolder).

- module 슬라이스만 (cell 전부 제외 — swelling 양성 0이라 타입 치팅 방지).
- set = splits.json (배터리 단위), class = 슬라이스별 swelling_flag (swell / nonswell).
- 입력 512 정사각 레터박스(종횡비 보존, squish 금지). 자연분포 — 복제/oversample 없음.
  (불균형 대응은 학습 단계에서 class weight/sampler로. ⚠️ 학습 transform은 crop이 아닌 Resize여야 함.)

출력: data/swell_cls/{train,val,test}/{swell,nonswell}/<path>.jpg
사용: python3 preprocessing/make_swelling_cls.py
"""
import json, os
from PIL import Image, ImageOps
from multiprocessing import Pool

SRC = "data/cropped"
DST = "data/swell_cls"
SIZE = 512

split = json.load(open("data/splits.json", encoding="utf-8"))

def build_tasks():
    tasks = []
    with open("data/labels.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["type"] != "module":          # module 전용
                continue
            core = r["battery_key"].split("_", 1)[1]
            s = split.get(core)
            if s is None:
                continue
            cls = "swell" if r.get("swelling_flag") else "nonswell"
            tasks.append((s, r["path"], cls))
    return tasks

def work(rec):
    s, path, cls = rec
    src = os.path.join(SRC, path)
    if not os.path.exists(src):
        return ("missing", path)
    dst = os.path.join(DST, s, cls, path.replace("/", "__"))
    if os.path.exists(dst):
        return ("skip", path)
    try:
        im = Image.open(src).convert("RGB")
        im = ImageOps.pad(im, (SIZE, SIZE), color=(0, 0, 0))   # 레터박스
        im.save(dst, quality=90)
        return ("ok", path)
    except Exception as e:
        return ("err:" + str(e), path)

def main():
    for s in ["train", "val", "test"]:
        for c in ["swell", "nonswell"]:
            os.makedirs(os.path.join(DST, s, c), exist_ok=True)
    tasks = build_tasks()
    print("총 task:", len(tasks))
    stats = {"ok": 0, "skip": 0, "missing": 0, "err": 0}
    with Pool() as p:
        for i, (status, _) in enumerate(p.imap_unordered(work, tasks, chunksize=200), 1):
            key = "err" if status.startswith("err") else status
            stats[key] = stats.get(key, 0) + 1
            if i % 20000 == 0:
                print("  %d/%d  %s" % (i, len(tasks), stats))
    print("완료:", stats)
    print("=== 폴더별 파일 수 ===")
    for s in ["train", "val", "test"]:
        for c in ["swell", "nonswell"]:
            d = os.path.join(DST, s, c)
            print("  %-5s/%-9s: %d" % (s, c, len(os.listdir(d))))

if __name__ == "__main__":
    main()
