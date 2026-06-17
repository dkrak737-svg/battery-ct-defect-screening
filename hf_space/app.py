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

import gradio as gr

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


def screen(files, form, battery_id):
    paths = _paths(files)
    if not paths:
        return "⚠️ CT 슬라이스 이미지를 업로드하세요", None, {}, ""

    slices = [{"idx": i, "name": os.path.basename(p), "img_path": p}
              for i, p in enumerate(paths)]
    bid = (battery_id or "uploaded").strip()
    result   = infer_battery(MODELS, slices, form, bid)     # 형태는 사용자가 선택
    decision = decide(result)
    report   = make_report(result, decision)
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
            form  = gr.Radio(["module", "cell"], value="module",
                             label="배터리 형태 (직접 선택)")
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
