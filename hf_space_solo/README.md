---
title: Battery CT Defect Screening (My Models)
emoji: 🔋
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 🔋 배터리 CT 결함 스크리닝 — 본인 학습 모델 버전

내가 직접 학습한 모델만으로 구성한 데모. **팀원 모델(검출 module/cell, swelling 5-fold)은 사용하지 않습니다.**

| 트랙 | 모델 | 비고 |
|---|---|---|
| 팽창(swelling) | **timm efficientnet_b0** (이진분류) | module 전용, 512 레터박스, 배터리 k=1 |
| 기공(porosity) | **YOLO11-seg** | 1024 타일(cell 4×), 마스크 오버레이 |
| 조치 지침 | RAG + Claude | swelling/porosity → IATA 규정 매핑·지침 |

## 사용법
1. 한 배터리의 **CT 슬라이스(크롭된 회색 이미지)** 업로드
2. **형태**(module/cell) **직접 선택**
3. **검사** → 🟢/🟡/🔴 판정 · 기공 마스크 · 보고서 · IATA 지침

> ⚠️ AI-Hub 원본 데이터는 라이선스상 포함하지 않습니다(업로드형).
> ⏳ 기공은 seg 타일링이라 슬라이스 수에 비례해 느립니다(CPU Space).
> ℹ️ 형태는 수동 선택(종횡비 자동판정은 축별 단면 차이로 부정확).

## 가중치
`ANTHROPIC_API_KEY`(선택) 설정 시 Claude가 한/영 조치 지침서를 생성합니다(미설정 시 규정 매핑 폴백).
모델: https://huggingface.co/dkrak737/battery-ct-defect-models (`swelling_best.pt`, `porosity_best.pt`)
코드: https://github.com/dkrak737-svg/battery-ct-defect-screening
