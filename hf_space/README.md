---
title: Battery CT Defect Screening
emoji: 🔋
colorFrom: gray
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 🔋 배터리 CT 내부결함 스크리닝 데모

배터리를 CT로 스캔한 슬라이스에서 **내부 결함(다공성·레진 오버플로·전역 팽창)** 을 탐지해
**🟢 이상없음 / 🟡 검토 필요 / 🔴 이상 있음** 으로 판정하고, **IATA 항공운송 규정** 기반 조치 지침(한/영)을 생성합니다.
목적은 **기내 화재(내부 단락) 예방** — recall(놓침 최소) 우선.

## 구성 (3트랙 + RAG)
- **검출(YOLO11 detect)**: 형태별 분리 — `module`(다공성+레진) / `cell`(다공성). 250px 타일 OR 집계.
- **팽창(YOLO11 classify)**: 5-fold 앙상블, 배터리 swelling 슬라이스 비율 > 0.1 이면 팽창.
- **다공성 정밀 위치(YOLO11 seg)**: 대표 슬라이스에 마스크 오버레이.
- **RAG**: 탐지 결함 → IATA D.06/SP A154 등 규정 매핑 → Claude 지침서.

## 사용법
1. 한 배터리의 **CT 슬라이스(크롭된 회색 이미지)** 여러 장 업로드
2. **형태**: `자동`(기본) 또는 수동(module/cell)
3. **검사** → 판정 · 오버레이 · 보고서 · 조치 지침

> ⚠️ AI-Hub 원본 데이터는 라이선스상 포함하지 않습니다. 본인 슬라이스를 업로드하세요.

### 형태 자동판정의 한계
자동판정은 슬라이스 **최대 종횡비**(cell≈13:1 / module≈2.4:1)를 쓰는 **이 데이터셋 기준 휴리스틱**입니다.
같은 배터리도 자른 축에 따라 단면 종횡비가 달라지고(cell의 가로 단면은 정사각), 다른 배터리 형태·CT 장비엔 일반화되지 않습니다.
모호하면(3~5:1) 경고를 띄우며, **수동 선택으로 override** 할 수 있습니다.
범용 해법은 **형태 분류기 학습** 또는 **스캔 메타데이터**에서 형태를 읽는 것(향후 과제).

## 환경변수 (Space Settings)
- `HF_MODEL_REPO` — 가중치 모델 레포 id (필수)
- `ANTHROPIC_API_KEY` — (Secret) RAG 실제 지침 생성용. 없으면 규정 매핑만 폴백.
- `GLUE_NO_SEG=1` — (선택) CPU Space 속도용, seg 마스크 끄고 박스만.

코드: https://github.com/dkrak737-svg/battery-ct-defect-screening
