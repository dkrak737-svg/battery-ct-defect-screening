# -*- coding: utf-8 -*-
"""
app.py (HF Space · 업로드형 데모)
사용자가 한 배터리의 CT 슬라이스(이미 크롭된 회색 이미지)를 업로드 → 추론 → 3존 판정 + 오버레이 + 보고서 + IATA 지침.

AI-Hub 원본 데이터는 배포하지 않는다(라이선스). 발표/시연 때 직접 슬라이스를 업로드해 사용.

환경변수:
  HF_MODEL_REPO       가중치 받을 HF 모델 레포 (예: user/battery-ct-defect-models)
  ANTHROPIC_API_KEY   (Space Secret) RAG 지침 생성용. 없으면 규정 매핑 폴백.
  GLUE_NO_SEG=1       seg 정밀 마스크 끄기(CPU Space 속도용; 박스만 표시)
"""
import os
import statistics

import gradio as gr
from PIL import Image

from load_models import load_all
from infer import infer_battery
from decide import decide
from render import make_overlay, make_report, pick_slice, recommend

MODELS = load_all()
USE_SEG = os.environ.get("GLUE_NO_SEG", "") == ""


def _paths(files):
    out = []
    for f in files or []:
        out.append(f if isinstance(f, str) else getattr(f, "name", None))
    return [p for p in out if p]


# 종횡비 자동 형태 판정: cell≈1:13(긴막대), module≈1:2.4(직사각).
#   ⚠️ 같은 배터리도 자른 축에 따라 단면 종횡비가 다름(cell도 가로 단면은 정사각).
#   → 슬라이스 '최대' 종횡비로 판정: cell은 어느 축에선 ~13:1이 나오고 module은 최대 ~2.4:1.
#   3~5 구간은 모호 → 수동 확인 권장. (단일·비대표 슬라이스만 올리면 빗나갈 수 있음)
FORM_RATIO_THR  = 5.0
FORM_GRAY_LOW   = 3.0


def detect_form(paths):
    """반환: (form, max_ratio, note)."""
    ratios = []
    for p in paths:
        try:
            w, h = Image.open(p).size
            lo, hi = min(w, h), max(w, h)
            if lo > 0:
                ratios.append(hi / lo)
        except Exception:
            pass
    if not ratios:
        return "module", None, "이미지 크기를 못 읽어 module 로 가정"
    r = max(ratios)                               # 가장 길쭉한 뷰 기준
    if r > FORM_RATIO_THR:
        return "cell", r, ""
    note = " ⚠️ 형태 모호 — 수동 확인 권장" if r >= FORM_GRAY_LOW else ""
    return "module", r, note


def screen(files, form, battery_id):
    paths = _paths(files)
    if not paths:
        return "⚠️ CT 슬라이스 이미지를 업로드하세요", None, {}, ""

    # 형태: '자동'이면 종횡비로 추정(이 데이터셋 휴리스틱), 아니면 사용자 선택값
    ratio = None; note = ""
    if form == "자동":
        form_used, ratio, note = detect_form(paths)
    else:
        form_used = form

    slices = [{"idx": i, "name": os.path.basename(p), "img_path": p}
              for i, p in enumerate(paths)]
    bid = (battery_id or "uploaded").strip()
    result   = infer_battery(MODELS, slices, form_used, bid)
    decision = decide(result)
    report   = make_report(result, decision)
    report["형태 판정"] = (f"{form_used} (자동 · 최대 종횡비 {ratio:.1f}:1{note})" if ratio is not None
                          else f"{form_used} (수동 선택)")
    guide    = recommend(result, decision)

    if decision["zone"].startswith("🟢"):
        return decision["zone"], None, report, guide
    cur, crop = pick_slice(result, slices)
    seg_model = MODELS["seg"] if USE_SEG else None
    overlay = make_overlay(crop, result, cur, seg_model) if crop is not None else None
    return decision["zone"], overlay, report, guide


with gr.Blocks(title="배터리 CT 결함 스크리닝") as demo:
    gr.Markdown(
        "## 🔋 배터리 CT 내부결함 스크리닝\n"
        "한 배터리의 **CT 슬라이스(크롭된 회색 이미지)** 를 업로드하면 "
        "다공성·레진·팽창을 탐지해 **🟢 이상없음 · 🟡 검토 필요 · 🔴 이상 있음** 으로 판정하고, "
        "IATA 항공운송 규정 기반 조치 지침을 생성합니다.\n"
        "> ⚠️ AI-Hub 원본 데이터는 포함하지 않습니다. 본인 슬라이스를 업로드하세요."
    )
    with gr.Row():
        with gr.Column():
            files = gr.File(file_count="multiple", file_types=["image"],
                            label="CT 슬라이스 업로드 (한 배터리)")
            form  = gr.Radio(["자동", "module", "cell"], value="자동",
                             label="배터리 형태")
            gr.Markdown(
                "<sub>자동 = 종횡비로 추정(이 AI-Hub pouch 데이터 기준 휴리스틱). "
                "다른 형태/장비면 빗나갈 수 있으니 **수동 선택으로 override** 하세요. "
                "범용 해법은 형태 분류기 학습 또는 스캔 메타데이터.</sub>"
            )
            bid   = gr.Textbox(label="배터리 ID (선택)", placeholder="예: module_402")
            btn   = gr.Button("검사", variant="primary")
        with gr.Column():
            verdict = gr.Label(label="판정")
            image   = gr.Image(label="위치 (다공성=마스크 · 레진=박스)")
    report = gr.JSON(label="보고서")
    guide  = gr.Markdown(label="조치 지침 (IATA · 한/영)")

    btn.click(screen, [files, form, bid], [verdict, image, report, guide])


if __name__ == "__main__":
    demo.launch()
