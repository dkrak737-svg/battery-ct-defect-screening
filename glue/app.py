# -*- coding: utf-8 -*-
"""
app.py
데모 UI (Gradio). 배터리 선택 → 추론 → 신호등 + 오버레이 + 보고서 + 조치 지침.

실행:
  cd glue
  export GLUE_DATA_ROOT=/workspace/battery-ct-security/data   # 원본(4000×4000 .jpg + .json) 루트
  python app.py        # RunPod 은 demo.launch(share=True) 또는 포트포워딩

흐름:  load_battery → infer_battery → decide → render(overlay/report/recommend)
"""
import gradio as gr

from load_models import load_all
from infer import infer_battery
from decide import decide
from render import make_overlay, make_report, pick_slice, recommend
from data import list_batteries, load_battery

MODELS = load_all()        # 시작할 때 딱 한 번만 로드


def screen(battery_id):
    slices, cell_type = load_battery(battery_id)
    result   = infer_battery(MODELS, slices, cell_type, battery_id)
    decision = decide(result)

    report = make_report(result, decision)
    guide  = recommend(result, decision)

    # 🟢 이상 없음 → 위치 이미지 없음
    if decision["zone"].startswith("🟢"):
        return decision["zone"], None, report, guide

    cur, crop = pick_slice(result, slices)        # 검출 위치 없으면(swelling 단독) None
    overlay = make_overlay(crop, result, cur, MODELS["seg"]) if crop is not None else None
    return decision["zone"], overlay, report, guide


with gr.Blocks(title="배터리 결함 스크리닝") as demo:
    gr.Markdown("## 배터리 결함 스크리닝 데모\n🟢 이상 없음 · 🟡 검토 필요 · 🔴 이상 있음")
    with gr.Row():
        bid = gr.Dropdown(choices=list_batteries(), label="배터리 선택")
        btn = gr.Button("검사", variant="primary")
    verdict = gr.Label(label="판정")
    image   = gr.Image(label="위치 (다공성=마스크 · 레진=박스)")
    report  = gr.JSON(label="보고서")
    guide   = gr.Markdown(label="조치 지침 (IATA · 한/영)")

    btn.click(screen, inputs=bid, outputs=[verdict, image, report, guide])


if __name__ == "__main__":
    demo.launch()
