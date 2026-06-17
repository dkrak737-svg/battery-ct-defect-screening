# -*- coding: utf-8 -*-
"""
app.py (solo Space · 본인 모델만)
swelling(timm) + porosity(seg) + RAG. 팀원 모델(검출 module/cell, swelling 5-fold) 미사용.

업로드형: 한 배터리의 CT 슬라이스(크롭된 회색 이미지)를 올리면 추론.
환경변수: HF_MODEL_REPO(가중치 레포), ANTHROPIC_API_KEY(선택, RAG 지침).
"""
import os

import gradio as gr

from load_models import load_all
from infer import infer_battery
from decide import decide
from render_solo import make_report, recommend, overlay

MODELS = load_all()


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
    guide = recommend(result, decision)
    img = overlay(result, slices)
    return decision["zone"], img, report, guide


with gr.Blocks(title="배터리 CT 결함 스크리닝 (본인 모델)") as demo:
    gr.Markdown(
        "## 🔋 배터리 CT 결함 스크리닝 — 본인 학습 모델 버전\n"
        "**swelling(timm efficientnet_b0) + porosity 기공(YOLO11-seg) + RAG** 만 사용. "
        "(팀원 검출·5-fold 모델 미포함)\n"
        "한 배터리의 **CT 슬라이스(크롭된 회색 이미지)** 를 업로드하세요.\n"
        "> ⚠️ AI-Hub 원본 데이터는 포함하지 않습니다. swelling 분류기는 module 전용입니다.\n"
        "> ⏳ porosity는 seg 타일링이라 슬라이스 수에 비례해 시간이 걸립니다(CPU)."
    )
    with gr.Row():
        with gr.Column():
            files = gr.File(file_count="multiple", file_types=["image"],
                            label="CT 슬라이스 업로드 (한 배터리)")
            form  = gr.Radio(["module", "cell"], value="module", label="배터리 형태 (직접 선택)")
            bid   = gr.Textbox(label="배터리 ID (선택)")
            btn   = gr.Button("검사", variant="primary")
        with gr.Column():
            verdict = gr.Label(label="판정")
            image   = gr.Image(label="기공 위치 (seg 마스크)")
    report = gr.JSON(label="보고서")
    guide  = gr.Markdown(label="조치 지침 (IATA · 한/영)")

    btn.click(screen, [files, form, bid], [verdict, image, report, guide])


if __name__ == "__main__":
    demo.launch()
