"""배터리 단위 층화 split 생성 -> splits.json (swelling / porosity 공용).

규칙 (2026-06-11 확정):
- 배터리 단위 80/10/10, seed 고정. 한 배터리의 모든 슬라이스는 한 세트에만(누수 금지).
- 층화 키 = (type, swelling_status, porosity, resin) 비례 분배.
- swelling_status: none(부풂 없음) / alltrue(전 슬라이스 true) / mixed(true & false 둘 다).
- 부풂 판정 = any-positive: 슬라이스 1장이라도 swelling_flag=True 이면 그 배터리는 부풂.
- mixed module(진짜 어려운 케이스)은 porosity 무시하고 단일 그룹으로 묶어 test에 최소 1개 보장.
- 작은 그룹 규칙: n>=10 -> round(0.1n) val/test; 3<=n<10 -> val1/test1; n==2 -> train1/test1; n==1 -> train.

사용: python3 preprocessing/make_splits.py
출력: data/splits.json  ({ "cell_pouch_0101": "train", ... })
"""
import json, collections, random

LABELS = "data/labels.jsonl"
OUT = "data/splits.json"
SEED = 42
random.seed(SEED)

# 1) 배터리별 메타 집계 (any-positive)
B = {}
with open(LABELS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        core = r["battery_key"].split("_", 1)[1]   # set 접두사 제거: cell_pouch_0101
        d = B.setdefault(core, {"type": r["type"], "po": False, "re": False, "sw_t": 0, "sw_f": 0})
        if r.get("swelling_flag"):
            d["sw_t"] += 1
        else:
            d["sw_f"] += 1
        for b in r["boxes"]:
            if b["cls"] == 0:
                d["po"] = True
            elif b["cls"] == 1:
                d["re"] = True

def sw_status(d):
    if d["sw_t"] == 0:
        return "none"
    if d["sw_f"] == 0:
        return "alltrue"
    return "mixed"

def group_key(d):
    s = sw_status(d)
    if s == "mixed":                       # 어려운 케이스: porosity 무시, test 보장 목적
        return (d["type"], "mixed")
    return (d["type"], s, "PO" if d["po"] else "-", "RE" if d["re"] else "-")

# 2) 그룹별 비례 분배
groups = collections.defaultdict(list)
for core, d in B.items():
    groups[group_key(d)].append(core)

split = {}
for key in sorted(groups, key=str):
    members = sorted(groups[key])          # 결정적 정렬 후 셔플
    random.shuffle(members)
    n = len(members)
    if n >= 10:
        n_test = round(n * 0.1); n_val = round(n * 0.1)
    elif n >= 3:
        n_test = 1; n_val = 1
    elif n == 2:
        n_test = 1; n_val = 0
    else:
        n_test = 0; n_val = 0
    test = members[:n_test]
    val = members[n_test:n_test + n_val]
    train = members[n_test + n_val:]
    for c in train: split[c] = "train"
    for c in val:   split[c] = "val"
    for c in test:  split[c] = "test"

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(split, f, indent=1, ensure_ascii=False)

# 3) 검증 출력
def is_neg_mod(d):  return d["type"] == "module" and d["sw_t"] == 0
def is_sw_mod(d):   return d["type"] == "module" and d["sw_t"] > 0
def is_mixed(d):    return d["type"] == "module" and d["sw_t"] > 0 and d["sw_f"] > 0

print("생성:", OUT, "| 배터리 총", len(split))
print()
print("set    n   | swell+module  neg-module  mixed | porosity  cell")
for s in ["train", "val", "test"]:
    cores = [c for c in split if split[c] == s]
    swpos = sum(1 for c in cores if is_sw_mod(B[c]))
    neg   = sum(1 for c in cores if is_neg_mod(B[c]))
    mix   = sum(1 for c in cores if is_mixed(B[c]))
    por   = sum(1 for c in cores if B[c]["po"])
    cell  = sum(1 for c in cores if B[c]["type"] == "cell")
    print("%-5s %3d  |    %3d         %3d       %3d  |   %3d    %3d" % (s, len(cores), swpos, neg, mix, por, cell))

print()
print("[보장 확인] mixed module 분배:")
for s in ["train", "val", "test"]:
    ms = sorted(c for c in split if split[c] == s and is_mixed(B[c]))
    print("  %-5s: %s" % (s, ms))
