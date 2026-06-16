# 글루(glue) — 세 트랙을 잇는 데모 파이프라인

팀원 모델(검출·swelling) + 내 모델(porosity seg) + RAG 를 하나의 데모로 잇는 **배관 코드**. 재학습 없음.

## 데이터 흐름
```
원본 CT 슬라이스 (4000×4000 .jpg + .json)
    -> data.load_battery()      배터리별 슬라이스+outline 로드
    -> infer.infer_battery()    세 트랙 추론 -> 표준 결과(BatteryResult)
    -> decide.decide()          3존 판정 🟢/🟡/🔴
    -> render.*                 대표 슬라이스 오버레이 + 보고서 + RAG 조치 지침
    -> app.py                   Gradio 화면
```

## 실제 받은 자료에 맞춘 구성 (★ 뼈대 초안과 다름)
| 트랙 | 모델 | task | 위치 |
|---|---|---|---|
| 검출(module) | `module_r01c.pt` (porosity=0, resin overflow=1) | detect | `teammate/weights/` |
| 검출(cell) | `cell_r06.pt` (porosity=0, **resin 없음**) | detect | `teammate/weights/` |
| swelling | `swell_kf0~4.pt` **5-fold 앙상블** (normal=0, swelling=1) | classify | `teammate/weights/` |
| porosity seg | `porosity_best.pt` (정밀 마스크 오버레이용) | segment | 루트 |
| RAG | `generate_guidance.py` + `index.npz` | IATA 지침 | `teammate/rag/` |

## 파일 역할
- `load_models.py` — 위 모델 한 번에 로드 (`{module, cell, swelling:[×5], seg}`)
- `recipe.py` — **팀원 전처리/추론 레시피 고정** (크롭·250타일·imgsz·conf·letterbox·임계값)
- `seg_tiles.py` — porosity seg 타일링(1024, cell 4×)+마스크 stitch (대표 슬라이스에만)
- `data.py` — 원본 루트 스캔, `list_batteries()` / `load_battery()` (슬라이스별 outline)
- `infer.py` — 배터리 → 표준 결과(BatteryResult). **이 형식이 모든 모듈의 약속**
- `decide.py` — 3존 규칙 (검출 OR + cell 검토 큐 + swelling 비율)
- `render.py` — 오버레이(다공성=마스크/레진=박스/팽창=없음) + 보고서 + RAG 조치 지침
- `app.py` — Gradio 데모

## 고정한 운영 레시피 (recipe.py / README_HANDOFF §3 일치)
- **검출**: outline 크롭(pad max(25,5%)) → 250×250 타일 overlap 0.2 → predict imgsz **module 512 / cell 640**, conf **0.05** → 타일 박스 OR 집계
- **swelling**: outline 크롭 → letterbox **224** → 5-fold 확률 평균 → 슬라이스 swelling 비율 **>0.1** 이면 배터리 swelling
- **seg**: cell 4× 확대 → 1024 타일 overlap 0.25 → 마스크를 크롭 좌표로 stitch

## 남은 연결 (운영 시)
1. `GLUE_DATA_ROOT` — 원본(4000×4000 .jpg + .json) 루트 경로 (env). 없으면 `/workspace/.../data`, `d:/CT DATA` 순으로 탐색.
2. `ANTHROPIC_API_KEY` — RAG 지침 생성용. `teammate/rag/.env` 에 입력. **없어도 데모는 동작**(결정적 매핑 조항만 표시로 폴백).
3. `GLUE_MAX_SLICES` (선택) — 배터리당 슬라이스 상한(데모 속도). 0=전량.
4. `decide.HIGH` — 검출 red/yellow 경계. grading sheet 로 튜닝.

## 검증 순서 (RunPod, GPU)
1. `python load_models.py` → 5종(+5fold) 열림 + 클래스 순서 확인
   (module `['porosity','resin overflow']`, swelling `{0:'normal',1:'swelling'}` 확인)
2. `export GLUE_DATA_ROOT=...` 후 `python -c "import data; print(data.list_batteries()[:5])"` → 배터리 목록 뜨는지
3. 배터리 1개로 `infer_battery` → 표준 결과 dict 모양/값 확인
4. 팀원이 결과 아는 배터리(예: module resin 402, swelling 311, cell 등)로 → 같은 플래그 나오는지 대조
5. `python app.py` → 화면에서 🟢/🟡/🔴 + 오버레이 + 보고서 + 지침 확인

## 업로드 전 (HF · Git) ⚠️
- API 키/토큰 스크럽 (`.env` 는 커밋 금지)
- AI-Hub 데이터 라이선스 — 원본 데이터는 올리지 말기
- 코드는 Git, 큰 가중치는 HF(LFS)
