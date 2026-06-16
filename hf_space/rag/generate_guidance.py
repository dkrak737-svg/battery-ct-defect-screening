# -*- coding: utf-8 -*-
"""탐지 결과(배터리별 결함) → 규정 검색(RAG) → Claude API 조치 지침서(한/영) 생성.

핵심 정책(사용자 지시):
  - 결함이 탐지된 배터리만 규정 기반 조치 지침서를 생성한다.
  - 결함이 없는 배터리는 RAG/생성 없이 "이상없음(정상, 운송 가능)"으로 표기한다.

입력(JSON, --input): [{"battery_id": "...", "form": "cell|module",
                       "defects": ["swelling","porosity","resin_overflow"]}]
  defects 빈 배열 = 이상없음.
출력: 콘솔 + report.md
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(HERE, "knowledge_base.json")
INDEX = os.path.join(HERE, "index.npz")
MODEL_EMB = "paraphrase-multilingual-MiniLM-L12-v2"


def load_env():
    """rag/.env 를 읽어 환경변수에 주입(이미 설정된 값은 유지)."""
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v and not os.environ.get(k):
            os.environ[k] = v


load_env()
MODEL_GEN = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")

NORMAL_KO = "이상없음 — 내부결함 미탐지. 운송 제한 사유 없음(UN38.3 통과·SoC 등 일반요건 충족 전제)."
NORMAL_EN = "No anomaly — no internal defect detected. No transport restriction applies (assuming UN38.3 pass, SoC, etc.)."


def load_kb():
    kb = json.load(open(KB, encoding="utf-8"))
    by_id = {c["id"]: c for c in kb["chunks"]}
    return kb, by_id


def retrieve(defects, kb, by_id, topk=5):
    """결정적 매핑(defect_to_chunk) + 임베딩 검색을 합쳐 관련 조항 청크 반환."""
    mapping = kb["meta"]["defect_to_chunk"]
    chunk_ids = []
    for d in defects:
        chunk_ids += mapping.get(d, [])
    # 임베딩 보조 검색 (인덱스 있을 때만)
    if os.path.exists(INDEX):
        try:
            from sentence_transformers import SentenceTransformer
            idx = np.load(INDEX, allow_pickle=True)
            ids, emb = list(idx["ids"]), idx["emb"]
            model = SentenceTransformer(MODEL_EMB)
            q = " ".join(defects) + " 리튬배터리 결함 운송 금지 short circuit thermal runaway defective"
            qv = model.encode([q], normalize_embeddings=True)[0]
            sims = emb @ qv
            for i in np.argsort(-sims)[:topk]:
                chunk_ids.append(ids[i])
        except Exception as e:
            print(f"[경고] 임베딩 검색 생략({e}); 결정적 매핑만 사용", file=sys.stderr)
    # 중복 제거(순서 유지)
    seen, ordered = set(), []
    for cid in chunk_ids:
        if cid not in seen and cid in by_id:
            seen.add(cid); ordered.append(by_id[cid])
    return ordered


def build_prompt(battery, chunks):
    refs = "\n\n".join(
        f"[{c['source']} · {c['ref']}]\n(KO) {c['text_ko']}\n(EN) {c['text_en']}"
        for c in chunks
    )
    return f"""당신은 항공 위험물(리튬배터리) 규정 준수 전문가입니다. CT 내부결함 AI가 탐지한 결함과 아래 규정 발췌만을 근거로, 한국어와 영어를 병기한 조치 지침서를 작성하세요. 규정에 없는 내용은 지어내지 말고, 각 판단마다 근거 조항(출처·조항번호)을 인용하세요.

[배터리]
- ID: {battery['battery_id']}
- 형태: {battery['form']}
- 탐지 결함: {', '.join(battery['defects'])}

[근거 규정 발췌]
{refs}

[작성 형식]
1. 판정 결론 (Verdict) — 운송 가능/금지 여부, 한/영
2. 결함별 위험 분석 (Hazard) — 탐지 결함이 규정상 어떤 위험에 해당하는지, 근거 조항 인용
3. 조치 지침 (Required Actions) — 운송 금지 시 처리 절차, 라벨/서류, 승인 필요 여부 등, 근거 조항 인용
4. 근거 조항 목록 (References)

간결하고 실무적으로. 각 항목 한국어 다음 영어."""


def generate(battery, chunks):
    import anthropic
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경변수 사용
    msg = client.messages.create(
        model=MODEL_GEN,
        max_tokens=4000,
        messages=[{"role": "user", "content": build_prompt(battery, chunks)}],
    )
    return msg.content[0].text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="탐지 결과 JSON 경로")
    ap.add_argument("--out", default=os.path.join(HERE, "report.md"))
    args = ap.parse_args()

    kb, by_id = load_kb()
    batteries = json.load(open(args.input, encoding="utf-8"))
    lines = ["# 배터리 항공운송 적합성 조치 지침서\n"]

    for b in batteries:
        defects = b.get("defects", [])
        lines.append(f"\n## 배터리 {b['battery_id']} ({b['form']})\n")
        if not defects:
            # 이상없음 — RAG/생성 없이 표기
            lines.append(f"**판정:** {NORMAL_KO}\n\n**Verdict:** {NORMAL_EN}\n")
            print(f"[{b['battery_id']}] 이상없음 → 지침서 생략")
            continue
        chunks = retrieve(defects, kb, by_id)
        print(f"[{b['battery_id']}] 결함 {defects} → 관련 조항 {len(chunks)}개 검색 → 지침서 생성")
        guidance = generate(b, chunks)
        lines.append(guidance + "\n")

    open(args.out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\n지침서 저장: {args.out}")


if __name__ == "__main__":
    main()
