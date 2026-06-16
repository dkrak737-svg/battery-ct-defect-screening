# 배터리 CT 내부결함 AI 탐지 — 인계 패키지

배터리 단독 CT 스캔 → 내부결함 AI 탐지 → IATA 규정 매칭 → 조치 지침서(한/영) 자동 생성.
목적: **기내 화재 방지**. 이 패키지로 **모델 추론 재현 + 재학습 + RAG 지침서 생성**이 가능합니다.

---

## 0. 환경 (재현 필수)

| 항목 | 값 |
|---|---|
| Python | **3.12.3** |
| ultralytics | **8.4.6x** (YOLO11) |
| torch | 2.8 (CUDA cu128) |
| GPU(학습) | RTX 5090 32GB |
| RAG 임베딩 | sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` (dim 384) |
| RAG 생성 | Anthropic Claude `claude-opus-4-8` |

```bash
pip install ultralytics==8.4.6 torch --index-url https://download.pytorch.org/whl/cu128
pip install sentence-transformers anthropic numpy pyyaml pillow
```

> ⚠️ **.pt만으로는 못 돌립니다.** 입력 크기·타일링·크롭 레시피가 전처리/평가 코드에 들어있습니다. 아래 conf·imgsz를 반드시 그대로 쓰세요.

---

## 1. 모델 3트랙 요약

| 트랙 | 형태 | 결함 | 가중치 | task | 비고 |
|---|---|---|---|---|---|
| **module** | module_pouch(1:2.4 직사각, 87개) | porosity, resin overflow | `weights/module_r01c.pt` | detect | **운영 가능** |
| **swelling** | module_pouch(67개) | swelling(전역 팽창) | `weights/swell_kf0~4.pt` (5-fold 앙상블) | classify | **운영 가능** |
| **cell** | cell_pouch(1:13 막대, 47개) | porosity | `weights/cell_r06.pt` | detect | 형태 한계 → **검토 큐** |

> 같은 porosity라도 형태별 변별력이 달라 **모델을 분리**(module 1:2.4는 맥락이 있어 성공, cell 1:13은 정상 세로구조와 구분 불가).

---

## 2. 클래스 목록·순서 (nc / names)

| 모델 | nc | names (인덱스 순) |
|---|---|---|
| **module** (detect) | 2 | `['porosity', 'resin overflow']` → **porosity=0, resin overflow=1** |
| **cell** (detect) | 1 | `['porosity']` → porosity=0 |
| **swelling** (classify) | 2 | `['normal', 'swelling']` → **normal=0, swelling=1** (폴더명 알파벳 순) |

→ `data_yaml/cell_battery_ct.yaml`, `data_yaml/module_battery_ct.yaml` 참고. swelling은 yaml 없이 `images/{train,val}/{normal,swelling}/` 폴더 구조.

---

## 3. 사용한 conf 임계값 (★ 운영값)

| 모델 | imgsz | conf | 집계 규칙 |
|---|---|---|---|
| **module** | **512** | **0.05** | 타일→슬라이스→배터리 OR (k=1, 발화 타일 1개↑면 불량) |
| **cell** | **640** | **0.05** | 동일 OR 집계. FP는 auto-reject 아니라 **검토 우선순위 큐** |
| **swelling** | **224** | (분류 top1) | 배터리별 swelling 슬라이스 비율 > **0.1** 이면 swelling 배터리 |

> conf는 학습이 아니라 **추론/평가 단계에서 낮춰** recall(놓침 0)을 확보합니다. best.pt는 mAP 기준 저장이지만 운영은 conf 0.05.
> 타일 mAP는 KPI로 쓰지 않음(작은 객체 IoU 민감 → 신기루). **진짜 KPI = 배터리/슬라이스 OR 집계**.

---

## 4. 학습 설정 (원본 args.yaml: `data_yaml/train_args_*.yaml`)

공통: optimizer `SGD`, seed 42, pretrained, `close_mosaic 10`, AMP.

| | module_r01c | cell_r06 | swell_kf0~4 |
|---|---|---|---|
| base | yolo11n.pt | yolo11n.pt | yolo11n-cls.pt |
| epochs / patience | 200 / 60 | 200 / 60 | 60 / 20 |
| imgsz | 512 | 640 | 224 |
| batch | 256 | 128 | 128 |
| lr0 | 0.003 | 0.003 | 0.001 |
| 증강 | 기본(mosaic/hsv) | 기본 | **회전·원근 OFF**, fliplr 0.5, hsv_v 0.3 (외형 단서 보존) |

---

## 5. 데이터 & split (재현용)

- **출처**: AI-Hub `103.배터리 불량 이미지 데이터` (datasetkey 71687), **CT만 사용**(RGB/Exterior 제외). 원본 4000×4000, 배터리가 이미지의 ~3.9%.
- **★ 분할은 배터리(form_num) 단위.** AI-Hub 기본 train/val은 같은 배터리를 슬라이스로 쪼개 **누수** → 폴더 무시하고 재분할(`preprocess_*.py`가 처리).
- **module resin stratified**: resin 보유 10개 배터리(402~411)를 train 8 / val 2(405,407)로 강제 배분(val resin 0개 방지).
- **swelling 5-fold**: 정상 배터리 20개 전부 + swelling 67개를 stratified k-fold. 라벨 정제(swelling 배터리는 swelling 단면만, 정상 배터리는 normal 단면만).

### 전처리 → 학습 → 평가 실행 예시
```bash
# (1) 전처리: 원본 → YOLO 타일 데이터셋
python code/preprocess_ct.py --src <원본루트> --form module_pouch --out data/yolo_module --min-box-px 11
python code/preprocess_ct.py --src <원본루트> --form cell_pouch   --out data/yolo_cell   --min-box-px 11
python code/preprocess_swelling.py --src <원본루트> --out data/yolo_swelling --n-folds 5 --fold-idx 0  # fold 0~4 반복

# (2) 학습
python code/train.py --data data/yolo_module/battery_ct.yaml --name module_r01c --imgsz 512 --batch 256 --epochs 200 --lr0 0.003
python code/train.py --data data/yolo_cell/battery_ct.yaml   --name cell_r06    --imgsz 640 --batch 128 --epochs 200 --lr0 0.003
python code/train_swelling.py --data data/yolo_swell_kf0/images --name swell_kf0 --imgsz 224 --batch 128  # fold 0~4

# (3) 평가 (진짜 KPI = 배터리 단위 OR 집계)
python code/eval_battery.py  --weights weights/module_r01c.pt --data data/yolo_module/battery_ct.yaml --imgsz 512 --confs 0.05
python code/eval_battery.py  --weights weights/cell_r06.pt    --data data/yolo_cell/battery_ct.yaml   --imgsz 640 --confs 0.05
python code/eval_swelling.py --weights weights/swell_kf0.pt   --data data/yolo_swelling/images        --imgsz 224 --batt-thr 0.1
```

---

## 6. 평가 수치 (배터리 단위 KPI = 운영 지표)

| 트랙 | 핵심 수치 | 평가 기준 |
|---|---|---|
| **swelling** | 배터리 recall **1.0**(단일) / **0.985**(5-fold), 정상 specificity **0.95**(정상 20개 전부), 슬라이스 swelling/normal recall 0.999/0.871 | 5-fold val, 배터리 단위 |
| **module** | 슬라이스 recall 0.998 / 슬라이스 **FP 0.002** / 배터리 **16/16** / mAP@0.5 0.948 (resin mAP 0.989, porosity 0.907) | conf 0.05, val 배터리 단위 |
| **cell** | 배터리 recall **1.0**(놓침 0) / 슬라이스 FP **0.21**(통짜 발화) | conf 0.05, baseline r06 |

- **3트랙 모두 결함 배터리 놓침 0** (배터리 recall 1.0 = 안전 목적 달성).
- **cell FP 0.21은 형태 물리한계** — porosity(폭 4~7px 세로선) vs 정상 내부 텍스처 = **0.91배(거의 동일)**, 깊이 균일. 6개 개선 실험(imgsz↑, 정상배경↑, 박스필터, yolo11s, 경계마스킹) 전부 실패로 한계 규명 → cell은 **검토 큐**(자동 reject 아님, recall 1.0 유지 + 사람이 우선순위 검토).

---

## 7. RAG / IATA 지침서

### 구성 (`rag/`)
| 파일 | 역할 |
|---|---|
| `knowledge_base.json` | 지식베이스 — **리튬배터리 항공운송 규정 10개 조항(한/영)**. c01 IATA D.06/SP A154(손상·결함 배터리 운송 금지), c02 국내 제7조, c03 제63조의2, c04 Class 9 UN3480/3481/3090/3091, c05 SoC 30% A331, c10 결함↔위험 매핑 등. `meta.defect_to_chunk`가 swelling/porosity/resin_overflow → 조항 ID 결정적 매핑 |
| `build_index.py` | kb → 임베딩 → `index.npz` (벡터스토어) |
| `generate_guidance.py` | 탐지결과 JSON → 규정 검색(결정적 매핑 + 임베딩 cosine) → Claude API 한/영 지침서 |
| `index.npz` | 사전 빌드된 임베딩(재빌드 불필요) |
| `sample_input.json` | 입력 예시 4개(결함 3 + 정상 1) |
| `.env.example` | **필요한 키 명세** (아래) |

### 필요한 키 (값은 별도 전달 — 이 패키지에 키 자체는 미포함)
- **`ANTHROPIC_API_KEY`** — Claude 지침서 생성용. `rag/.env.example`를 `rag/.env`로 복사 후 본인 키 입력.
- (선택) `ANTHROPIC_MODEL` — 기본 `claude-opus-4-8`.

### 실행
```bash
cp rag/.env.example rag/.env    # 키 입력
python rag/build_index.py       # 최초 1회 (index.npz 있으면 생략 가능)
python rag/generate_guidance.py --input rag/sample_input.json --out rag/report.md
```

### 입력 형식 / 정책
```json
[{"battery_id": "module_402", "form": "module", "defects": ["porosity","resin_overflow"]},
 {"battery_id": "module_500", "form": "module", "defects": []}]
```
- **★ 결함이 탐지된 배터리만 규정 인용 지침서를 생성**(D.06/A154/제7조/제63조의2 등).
- **결함 없으면(`defects: []`) "이상없음"으로 표기**하고 지침서 생략(사용자 지시 반영).
- 환각 방지: 미탐지 결함의 위험은 적용하지 않음. "운송 금지 건엔 SoC/UN38.3 면제경로 부적용" 법리 반영. 한/영 병기.

### 결함명 ↔ 위험 매핑
| 탐지 결함 | 규정상 위험 |
|---|---|
| swelling | 위험한 발열(dangerous evolution of heat) |
| porosity | 단락(short circuit) |
| resin overflow | 결함(defective) |
→ 셋 다 IATA D.06 / SP A154 상 **항공운송 금지**(damaged/defective 배터리).

---

## 8. 디렉토리

```
handoff/
├─ README_HANDOFF.md          ← 이 문서
├─ weights/                   ← 가중치 7종 (.pt)
│   ├─ module_r01c.pt  cell_r06.pt
│   └─ swell_kf0~4.pt (5-fold 앙상블; 단일 사용 시 1개만)
├─ code/                      ← 전처리·학습·평가 코드
├─ data_yaml/                 ← data.yaml 2종 + 학습 args.yaml 3종(설정 원본)
└─ rag/                       ← RAG 코드·지식베이스·인덱스·.env.example
```

문의: 전체 개발 경위·실험 로그는 별도 `PROGRESS.md` / `results/전체결과_종합.md` 참고.
