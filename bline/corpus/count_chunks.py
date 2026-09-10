#!/usr/bin/env python3
"""W3-P1 chunk复算：用冻结rag_demo.py的parse_docs/chunk_text原文逻辑，
对 docs/（冻结）+ bline/corpus/（新增）分别切分计数。
只import不修改基线；PYTHONDONTWRITEBYTECODE=1 运行，不在基线目录留字节码。
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)  # bline/
ROOT = os.path.dirname(BASE)  # rag_min_demo/
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location(
    "rag_demo_frozen", os.path.join(ROOT, "rag_demo.py"))
rag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag)

rows = []  # (src, chunk_id, doc_id, chars)
for src, d in (("docs", os.path.join(ROOT, "docs")),
               ("corpus", HERE)):
    for doc in rag.parse_docs(d):
        for j, c in enumerate(rag.chunk_text(doc["text"])):
            rows.append((src, f"{doc['doc_id']}#c{j}", doc["doc_id"], len(c)))

nd = sum(1 for r in rows if r[0] == "docs")
nc = sum(1 for r in rows if r[0] == "corpus")
print(f"docs chunks={nd} corpus chunks={nc} total={len(rows)}")
print(f"params MAX_CHARS={rag.MAX_CHARS} OVERLAP={rag.OVERLAP}")
for src, cid, did, ch in rows:
    print(f"{src:6s} {cid:22s} doc={did:18s} chars={ch}")
sys.exit(0 if len(rows) >= 30 else 1)
