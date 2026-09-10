#!/usr/bin/env python3
"""P3B-rest独立复核：逐行重算320探针7格点，与结果文件逐格比对。
用法：PYTHONDONTWRITEBYTECODE=1 python3 rag_min_demo/bline/w3_p3b_rest_verify.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rag_demo
from w1_hybrid import rrf_fusion, weighted_fusion
from w3_harness import build_extended, norm_item

with open(os.path.join(HERE, "w1_config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
with open(os.path.join(HERE, "w3_golden_v3.json"), encoding="utf-8") as f:
    v3 = json.load(f)
assert v3["version"] == "w3-v3-44"
with open(os.path.join(HERE, "w3_p3b_rest_results.json"), encoding="utf-8") as f:
    saved = json.load(f)

chunks0, index0 = build_extended()
assert len(chunks0) == 35
by_id = {c["chunk_id"]: c for c in chunks0}
seeds = [q for q in v3["queries"] if "trap" not in q]
assert len(seeds) == 32
NAMES = ["w_cos=0.0", "w_cos=0.2", "w_cos=0.4", "w_cos=0.6", "w_cos=0.8",
         "w_cos=1.0", "rrf_k=60"]
REF = "w_cos=0.6"


def score_with(st, chunks, index, query, top_k):
    qtoks = rag_demo.bigrams(query)
    qvec = index._vec(qtoks)
    if not any(qvec.values()):
        return []
    cos = {i: s for s, i in index.cosine(qvec)}
    bm = {i: s for s, i in index.bm25(qtoks)}
    if st[0] == "weighted":
        ranked = weighted_fusion(cos, bm, w_cos=st[1],
                                 candidates=st[2], top_k=top_k)
    else:
        ranked = rrf_fusion(cos, bm, k=st[1],
                            candidates=st[2], top_k=top_k)
    return [(s, chunks[i]) for s, i in ranked]


def grid(cands, chunks, index, qtext, doc, chs, top_k):
    out = {}
    for w in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        res = score_with(("weighted", w, cands), chunks, index, qtext, top_k)
        out[f"w_cos={w}"] = next((i + 1 for i, (_, c) in enumerate(res)
                                  if c["doc_id"] == doc), None)
    res = score_with(("rrf", cfg["rrf_k"], cands), chunks, index, qtext, top_k)
    out["rrf_k=60"] = next((i + 1 for i, (_, c) in enumerate(res)
                            if c["doc_id"] == doc), None)
    return out


def n_del10(q):
    return "".join(ch for i, ch in enumerate(q) if (i + 1) % 10 != 0)

def n_swap(q):
    a = list(q)
    for i in range(0, len(a) - 1, 2):
        a[i], a[i + 1] = a[i + 1], a[i]
    return "".join(a)

def n_trunc(q):
    return q[:len(q) // 2]


fails, checked = [], 0
# depth
for r in saved["depth_rows"]:
    item = next(q for q in seeds if q["q"] == r["seed_q"])
    doc, chs = norm_item(item)
    cands = {"C5": 5, "C10": 10, "C20": 20, "C35": 35}.get(r["variant"], cfg["candidates"])
    got = grid(cands, chunks0, index0, item["q"], doc, chs, r["top_k"])
    checked += 1
    if got != r["doc_ranks"]:
        fails.append(("depth", r["variant"], r["seed_q"][:30], got, r["doc_ranks"]))
# noise
for r in saved["noise_rows"]:
    item = next(q for q in seeds if q["q"] == r["seed_q"])
    doc, chs = norm_item(item)
    fn = {"N-del10": n_del10, "N-swap": n_swap, "N-trunc": n_trunc}[r["variant"]]
    assert fn(item["q"]) == r["qtext"], ("noise文本漂移", r["variant"])
    got = grid(cfg["candidates"], chunks0, index0, r["qtext"], doc, chs, cfg["top_k"])
    checked += 1
    if got != r["doc_ranks"]:
        fails.append(("noise", r["variant"], r["seed_q"][:30], got, r["doc_ranks"]))
# neardup
for r in saved["near_rows"]:
    item = next(q for q in seeds if q["q"] == r["seed_q"])
    doc, chs = norm_item(item)
    gold = by_id[chs[0]]["text"]
    n = len(gold)
    cuts = sorted(set([n // 4, n // 2, 3 * n // 4]))
    synth = [{"chunk_id": f"__interf__#s{k}", "doc_id": "__interf__",
              "title": "__interf__", "text": gold[:p] + gold[p + 1:],
              "chars": len(gold) - 1}
             for k, p in enumerate(cuts) if 0 < p < n]
    assert len(synth) == r["n_synth"], ("合成数漂移", r["seed_q"][:30])
    chunks = chunks0 + synth
    toks = [rag_demo.bigrams(c["text"]) for c in chunks]
    rag_demo.chunk_tokens_global = toks
    index = rag_demo.TfidfIndex(toks)
    got = grid(cfg["candidates"], chunks, index, item["q"], doc, chs, cfg["top_k"])
    checked += 1
    if got != r["doc_ranks"]:
        fails.append(("neardup", r["variant"], r["seed_q"][:30], got, r["doc_ranks"]))
rag_demo.chunk_tokens_global = [rag_demo.bigrams(c["text"]) for c in chunks0]

# 汇总数复算
for tag, key in (("depth", "depth_rows"), ("noise", "noise_rows"),
                 ("neardup", "near_rows")):
    rows = saved[key]
    hd = sum(1 for r in rows if r["hit_div"])
    rd = sum(1 for r in rows if r["rank_div"])
    exp = saved[{"depth": "depth", "noise": "noise",
                 "neardup": "neardup"}[tag]]
    assert hd == exp["hitdiv"] and rd == exp["rankdiv"] and len(rows) == exp["n"], \
        (tag, hd, rd, len(rows), exp)

print(f"[verify] 逐行重算 {checked} 行，分歧={len(fails)}")
for f in fails[:10]:
    print("  FAIL", f)
assert not fails, f"{len(fails)} 行复算不一致"
print("[verify] PASS：320行逐格一致，汇总数一致，noise变换文本一致，合成数一致")
