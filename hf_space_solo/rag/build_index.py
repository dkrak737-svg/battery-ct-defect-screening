# -*- coding: utf-8 -*-
"""지식베이스(knowledge_base.json) → 임베딩 인덱스(index.npz) 생성.
한국어 지원 multilingual 임베딩 모델 사용. 리튬배터리 조항만이라 코퍼스가 작아 CPU로 수초."""
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(HERE, "knowledge_base.json")
OUT = os.path.join(HERE, "index.npz")
MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 한/영 동시 지원, 경량


def chunk_text(c):
    # 한국어+영어를 한 임베딩에 합쳐 양쪽 질의 모두 매칭되게
    return f"{c['title_ko']} {c['title_en']}\n{c['text_ko']}\n{c['text_en']}\n태그:{' '.join(c['tags'])}"


def main():
    kb = json.load(open(KB, encoding="utf-8"))
    chunks = kb["chunks"]
    model = SentenceTransformer(MODEL)
    texts = [chunk_text(c) for c in chunks]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    ids = [c["id"] for c in chunks]
    np.savez(OUT, ids=np.array(ids), emb=emb.astype(np.float32))
    print(f"인덱스 저장: {OUT} ({len(ids)} 청크, dim={emb.shape[1]})")


if __name__ == "__main__":
    main()
