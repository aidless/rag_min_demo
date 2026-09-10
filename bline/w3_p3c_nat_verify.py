#!/usr/bin/env python3
"""P3C独立复核：逐候选×7格点重算（doc/chunk rank），核对汇总数与文本一致性。
用法：PYTHONDONTWRITEBYTECODE=1 python3 rag_min_demo/bline/w3_p3c_nat_verify.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rag_demo
from w1_hybrid import rrf_fusion, weighted_fusion
from w3_harness import build_extended, strategies

cfg = json.load(open(os.path.join(HERE, "w1_config.json"), encoding="utf-8"))
res = json.load(open(os.path.join(HERE, "w3_p3c_nat_results.json"),
                     encoding="utf-8"))
chunks, index = build_extended()
assert len(chunks) == 35
sts, names = strategies(cfg)
assert len(sts) == 7
REF = "w_cos=0.6"
by_id = {c["chunk_id"]: c for c in chunks}

ok = True
for r in res["rows"]:
    # 标注有效性：expect chunk存在且属expect doc
    for ch in r["expect_chunks"]:
        assert ch in by_id, (r["cid"], ch)
        assert by_id[ch]["doc_id"] == r["expect_doc"], (r["cid"], ch)
    for st, name in zip(sts, names):
        qtoks = rag_demo.bigrams(r["q"])
        qvec = index._vec(qtoks)
        assert any(qvec.values()), (r["cid"], "empty qvec")
        cos = {i: s for s, i in index.cosine(qvec)}
        bm = {i: s for s, i in index.bm25(qtoks)}
        if st[0] == "weighted":
            ranked = weighted_fusion(cos, bm, w_cos=st[1],
                                     candidates=st[2], top_k=cfg["top_k"])
        else:
            ranked = rrf_fusion(cos, bm, k=st[1],
                                candidates=st[2], top_k=cfg["top_k"])
        got = [(s, chunks[i]) for s, i in ranked]
        d = next((i + 1 for i, (_, c) in enumerate(got)
                  if c["doc_id"] == r["expect_doc"]), None)
        kk = next((i + 1 for i, (_, c) in enumerate(got)
                   if c["chunk_id"] in r["expect_chunks"]), None)
        if d != r["doc_ranks"][name] or kk != r["chunk_ranks"][name]:
            print(f"MISMATCH {r['cid']} {name}: doc {d}!={r['doc_ranks'][name]}, "
                  f"chunk {kk}!={r['chunk_ranks'][name]}")
            ok = False

hd = sum(1 for r in res["rows"] if r["hit_div"])
rd = sum(1 for r in res["rows"] if r["rank_div"])
assert hd == res["hitdiv"] and rd == res["rankdiv"], (hd, rd)
print(f"[P3C-verify] rows={len(res['rows'])} hitdiv={hd} rankdiv={rd} "
      f"逐格一致={'PASS' if ok else 'FAIL'}")
out = {"rows": len(res["rows"]), "hitdiv": hd, "rankdiv": rd,
       "cell_match": ok, "summary_match": True,
       "verdict": "PASS" if ok else "FAIL"}
with open(os.path.join(HERE, "w3_p3c_nat_verify.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
sys.exit(0 if ok else 1)
