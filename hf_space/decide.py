# -*- coding: utf-8 -*-
"""
decide.py
표준 결과(BatteryResult) → 3존 판정. 순수 규칙, 재학습 없음.

3존:  🟢 이상 없음  /  🟡 검토 필요  /  🔴 이상 있음
원칙: recall-first — 놓치는 것보다 과하게 거르는 게 낫다.

판정 근거(팀원 운영값):
  - 검출은 conf 0.05 OR 집계로 이미 '발화=결함'. 여기선 conf 로 red/yellow 만 가른다.
  - cell porosity 는 물리한계(FP 0.21) → 자동 reject 금지, 항상 '검토 큐'(yellow).
  - module 검출/swelling 은 신뢰 높음(recall~1.0, FP~0) → 확신 conf 면 red.
  - swelling 은 배터리 비율(>0.1)로 이미 판정됨 → 발화 시 red.
"""

# 검출 conf 이 이 값 이상이면 '확정'(red), 운영임계(0.05)~HIGH 면 '약한 신호'(yellow).
HIGH = 0.50

# 결함 내부명 → RAG/표시용 정식명
DEFECT_CANON = {"porosity": "porosity", "resin": "resin_overflow", "swelling": "swelling"}


def decide(result):
    """반환: {zone, defects, red, yellow, worst_conf, reasons}."""
    ct = result["cell_type"]
    red, yellow, reasons = [], [], []

    # (1) swelling — module 전용, 비율로 이미 배터리 판정됨
    sw = result["swelling"]
    if sw["flag"]:
        red.append("swelling")
        reasons.append(f"swelling 슬라이스 비율 {sw['ratio']*100:.0f}% (>10%) → 전역 팽창")

    # (2) 검출 결함 (porosity / resin) — 형태별 신뢰도로 zone 분기
    for d in ("porosity", "resin"):
        info = result[d]
        if not info["flag"]:
            continue
        cf = info["conf"]
        if ct == "cell":
            yellow.append(d)
            reasons.append(f"{d} 검출 (cell 물리한계 → 검토 큐, conf {cf:.2f})")
        elif cf >= HIGH:
            red.append(d)
            reasons.append(f"{d} 확정 (conf {cf:.2f})")
        else:
            yellow.append(d)
            reasons.append(f"{d} 약한 신호 (conf {cf:.2f})")

    # (3) 존 결정 — 나쁜 쪽 우선
    if red:
        zone = "🔴 이상 있음"
    elif yellow:
        zone = "🟡 검토 필요"
    else:
        zone = "🟢 이상 없음"
        reasons.append("세 결함 모두 미탐지")

    return {
        "zone":       zone,
        "defects":    red + yellow,                 # 내부명(porosity/resin/swelling)
        "red":        red,
        "yellow":     yellow,
        "worst_conf": max(result["porosity"]["conf"], result["resin"]["conf"], sw["conf"]),
        "reasons":    reasons,
    }
