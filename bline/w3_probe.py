#!/usr/bin/env python3
"""W3-P3 探针：只读复用冻结基线，输出扩展管线35 chunk全文 + 候选对抗题逐策略排名。
只读，不改基线；PYTHONDONTWRITEBYTECODE=1 运行。
用法：
  python3 rag_min_demo/bline/w3_probe.py --dump                # 打印35 chunk全文
  python3 rag_min_demo/bline/w3_probe.py --cands file.json     # 候选题逐策略doc/chunk排名
cands格式：[{"q": "...", "expect_doc": "x", "expect_chunks": ["x#c0"]}, ...]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rag_demo
from w1_hybrid import rrf_fusion, weighted_fusion
from w3_harness import build_extended, score_for

with open(os.path.join(HERE, "w1_config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
STS = [("weighted", w, cfg["candidates"]) for w in cfg["w_cos_grid"]]
STS.append(("rrf", cfg["rrf_k"], cfg["candidates"]))
NAMES = [f"w_cos={s[1]}" if s[0] == "weighted" else f"rrf_k={s[1]}" for s in STS]

chunks, index = build_extended()
print(f"extended chunks={len(chunks)}")

if "--dump" in sys.argv:
    for c in chunks:
        print("=" * 70)
        print(f"[{c['chunk_id']}] doc={c['doc_id']} chars={c['chars']}")
        print(c["text"])
elif "--duel" in sys.argv:
    p = sys.argv[sys.argv.index("--duel") + 1]
    with open(p, encoding="utf-8") as f:
        cands = json.load(f)
    for item in cands:
        print("=" * 70)
        print(f"Q: {item['q']}")
        qtoks = rag_demo.bigrams(item["q"])
        qvec = index._vec(qtoks)
        cos = sorted(index.cosine(qvec), reverse=True)[:5]
        bm = sorted(index.bm25(qtoks), reverse=True)[:5]
        print("  纯cos路top5:", [(chunks[i]["chunk_id"], round(s, 4))
                                for s, i in cos])
        print("  纯BM25路top5:", [(chunks[i]["chunk_id"], round(s, 4))
                                  for s, i in bm])
elif "--duelscan" in sys.argv:
    p = sys.argv[sys.argv.index("--duelscan") + 1]
    with open(p, encoding="utf-8") as f:
        items = json.load(f)
    qs = items["queries"] if isinstance(items, dict) else items
    for item in qs:
        q = item["q"] if isinstance(item, dict) else item
        qtoks = rag_demo.bigrams(q)
        qvec = index._vec(qtoks)
        cos = sorted(index.cosine(qvec), reverse=True)[:3]
        bm = sorted(index.bm25(qtoks), reverse=True)[:3]
        ct = [(chunks[i]["chunk_id"], round(s, 4)) for s, i in cos]
        bt = [(chunks[i]["chunk_id"], round(s, 4)) for s, i in bm]
        flip = "FLIP" if ct[0][0] != bt[0][0] else "same"
        mgn = (round(ct[0][1] / ct[1][1], 2) if len(ct) > 1 and ct[1][1]
               else float("inf"))
        print(f"[{flip} cos_margin={mgn}] {q}\n   cos={ct}\n   bm ={bt}")
elif "--cands" in sys.argv:
    p = sys.argv[sys.argv.index("--cands") + 1]
    with open(p, encoding="utf-8") as f:
        cands = json.load(f)
    for item in cands:
        doc, chs = item["expect_doc"], item.get("expect_chunks") or []
        print("=" * 70)
        print(f"Q: {item['q']}\n   expect_doc={doc} expect_chunks={chs}")
        dr, cr = [], []
        for st, name in zip(STS, NAMES):
            res = score_for(st, chunks, index, item["q"], cfg["top_k"])
            d = next((i + 1 for i, (_, c) in enumerate(res)
                      if c["doc_id"] == doc), None)
            k = next((i + 1 for i, (_, c) in enumerate(res)
                      if c["chunk_id"] in chs), None) if chs else None
            dr.append(d)
            cr.append(k)
            top = [(c["chunk_id"], round(s, 3)) for s, c in res]
            print(f"  {name:12s} doc_rank={d} chunk_rank={k} top3={top}")
        print(f"  doc分歧hit={sum(1 for d in dr if (d is None))} "
              f"rank集={sorted(set(str(d) for d in dr))}")
        print(f"  chunk分歧hit={sum(1 for k in cr if (k is None))} "
              f"rank集={sorted(set(str(k) for k in cr))}")
