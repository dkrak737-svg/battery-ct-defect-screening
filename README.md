# 🔋 배터리 CT 내부결함 스크리닝

배터리를 CT로 스캔한 슬라이스에서 **내부 결함(다공성 · 레진 오버플로 · 전역 팽창)** 을 탐지해
**🟢 이상없음 / 🟡 검토 필요 / 🔴 이상 있음** 으로 판정하고, **IATA 항공운송 규정** 기반 조치 지침(한/영)을 생성합니다.
목적은 **기내 화재(내부 단락) 예방** — 놓침 최소(recall) 우선.

## 🔗 링크
- **데모 (HF Space):** https://huggingface.co/spaces/dkrak737/battery-ct-defect-screening
- **가중치 (HF Model):** https://huggingface.co/dkrak737/battery-ct-defect-models

## 구성 (모델 3트랙 + RAG)
| 트랙 | 모델 | 역할 |
|---|---|---|
| 검출 | YOLO11 detect ×2 (형태별 분리: `module`=다공성+레진 / `cell`=다공성) | 250px 타일 → 슬라이스 → 배터리 OR 집계 |
| 팽창 | YOLO11 classify (5-fold 앙상블) | 배터리 swelling 슬라이스 비율 > 0.1 이면 팽창 |
| 다공성 정밀 위치 | YOLO11 seg | 대표 슬라이스 마스크 오버레이 |
| 조치 지침 | RAG + Claude | 탐지 결함 → IATA D.06/SP A154 등 매핑 → 지침서 |

## 파이프라인
```
CT 슬라이스 → 추론(3트랙) → 표준 결과(BatteryResult) → 3존 판정 → 오버레이 + 보고서 + IATA 지침
```

## 레포 구조
- `glue/` — 3트랙 + RAG 통합 데모(로컬용). 전처리/추론 레시피를 `recipe.py`에 고정
- `hf_space/` — HF Space 배포본(업로드형 데모, 가중치는 HF 모델 레포에서 로드)
- `teammate/` — 검출·팽창 모델 코드 + RAG 핸드오프
- `preprocessing/` · `vision/` — 전처리 · 타일링 · 학습 · 평가
- `config.py` · `splits.json` — 설정 · 배터리 단위 분할

## 데이터
AI-Hub `103.배터리 불량 이미지 데이터`(datasetkey 71687), **CT만 사용**. 배터리 단위 분할(누수 방지).
> ⚠️ 원본 데이터는 라이선스상 **재배포하지 않습니다.** 데모는 업로드형으로 동작합니다.

## 한계 / 향후
- Space의 형태 자동판정(종횡비)은 **이 데이터셋 휴리스틱** — 다른 형태/장비엔 수동 선택 또는 형태 분류기 학습 필요
- cell 다공성은 정상 텍스처와 물리적으로 유사 → 자동 reject 대신 **사람 검토 큐**
