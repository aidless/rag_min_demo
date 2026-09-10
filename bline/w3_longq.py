#!/usr/bin/env python3
"""W3-P3B long_query挖掘：长查询/段落级查询是否使7格点分歧。

机制假设（见rag_demo.py实测代码）：
- cosine路：query向量经TF-IDF log加权后L2归一化，长查询把权重质量摊薄；
- BM25路：对query二元组只按类型求和（qtf未参与），无query归一化，
  长查询的填充词匹配可无限累加。
两路共享cos top-candidates池（candidates=10），分歧只能来自池内重排；
本脚本全部用真实语料原文构造长查询，不编造文本：
  L-ceil      q + 金chunk原文（天花板：金答案逐字在查询里）
  L-fill-near q + 短问下ref首中非金chunk原文（最难填充压力）
  L-fill-far  q + fitness干扰文档原文（已知远域干扰项，P1）
  L-both      q + 金chunk原文 + 难干扰chunk原文（双粘贴）
口径：7格点（沿用w1_config），top_k=3；只记录分歧，不污染v3 golden。
这些是机制探针（含答案泄露），不是自然用户问题，结论只用于“权重是否有信号”。
用法：PYTHONDONTWRITEBYTECODE=1 python3 rag_min_demo/bline/w3_longq.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rag_demo
from w3_harness import build_extended, norm_item, score_for, strategies

with open(os.path.join(HERE, "w1_config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
with open(os.path.join(HERE, "w3_golden_v3.json"), encoding="utf-8") as f:
    v3 = json.load(f)
assert v3["version"] == "w3-v3-44", v3["version"]
seeds = [q for q in v3["queries"] if "trap" not in q]
assert len(seeds) == 32, len(seeds)

chunks, index = build_extended()
assert len(chunks) == 35, len(chunks)
sts, names = strategies(cfg)
assert len(sts) == 7, len(sts)
REF = "w_cos=0.6"
ref_st = sts[names.index(REF)]
by_id = {c["chunk_id"]: (i, c) for i, c in enumerate(chunks)}
fit = [c for c in chunks if c["doc_id"] == "fitness"]
assert fit, "fitness干扰chunk缺失"
FIT_TEXT = fit[0]["text"]


def pure_tops(query, k=3):
    qtoks = rag_demo.bigrams(query)
    qvec = index._vec(qtoks)
    cos = sorted(index.cosine(qvec), reverse=True)[:k]
    bm = sorted(index.bm25(qtoks), reverse=True)[:k]
    return ([chunks[i]["chunk_id"] for _, i in cos],
            [chunks[i]["chunk_id"] for _, i in bm])


def short_ref_top1_nongold(item, doc, chs):
    res = score_for(ref_st, chunks, index, item["q"], cfg["top_k"])
    for _, c in res:
        if c["doc_id"] != doc:
            return c["text"]
    # ref首中全是金：退回最长非金chunk
    cands = [c for c in chunks if c["doc_id"] != doc]
    return max(cands, key=lambda c: c["chars"])["text"]


def run_variant(qtext, doc, chs):
    rec = {"doc_rank": {}, "chunk_rank": {}}
    for st, name in zip(sts, names):
        res = score_for(st, chunks, index, qtext, cfg["top_k"])
        d = next((i + 1 for i, (_, c) in enumerate(res)
                  if c["doc_id"] == doc), None)
        k = next((i + 1 for i, (_, c) in enumerate(res)
                  if chs and c["chunk_id"] in chs), None)
        rec["doc_rank"][name] = d
        rec["chunk_rank"][name] = k
    return rec


rows = []
for item in seeds:
    doc, chs = norm_item(item)
    gold_text = by_id[chs[0]][1]["text"] if chs else None
    near_text = short_ref_top1_nongold(item, doc, chs)
    variants = {"L-fill-near": item["q"] + "\n" + near_text,
                "L-fill-far": item["q"] + "\n" + FIT_TEXT}
    if gold_text:
        variants["L-ceil"] = item["q"] + "\n" + gold_text
        variants["L-both"] = item["q"] + "\n" + gold_text + "\n" + near_text
    for vname, qtext in variants.items():
        cos3, bm3 = pure_tops(qtext)
        rec = run_variant(qtext, doc, chs)
        dh = set(str(v) for v in rec["doc_rank"].values())
        ch = set(str(v) for v in rec["chunk_rank"].values())
        ref_d = rec["doc_rank"][REF]
        hit_div = sum(1 for v in rec["doc_rank"].values()
                      if (v is None) != (ref_d is None))
        rows.append({"seed_q": item["q"], "variant": vname,
                     "expect_doc": doc, "expect_chunks": chs,
                     "qlen": len(qtext),
                     "pure_cos_top1": cos3[0], "pure_bm25_top1": bm3[0],
                     "pure_flip": cos3[0] != bm3[0],
                     "doc_ranks": rec["doc_rank"],
                     "chunk_ranks": rec["chunk_rank"],
                     "n_rank_values_doc": len(dh),
                     "n_rank_values_chunk": len(ch),
                     "hit_div_doc_vs_ref": hit_div})

n_flip = sum(1 for r in rows if r["pure_flip"])
n_rankdiv = sum(1 for r in rows if r["n_rank_values_doc"] > 1)
n_hitdiv = sum(1 for r in rows if r["hit_div_doc_vs_ref"] > 0)
print(f"[P3B] 探针={len(rows)} 纯路top1翻转={n_flip} "
      f"7格点rank分歧={n_rankdiv} hit分歧={n_hitdiv}")
for v in ("L-ceil", "L-fill-near", "L-fill-far", "L-both"):
    sub = [r for r in rows if r["variant"] == v]
    print(f"  {v:12s} n={len(sub)} flip={sum(1 for r in sub if r['pure_flip'])} "
          f"rankdiv={sum(1 for r in sub if r['n_rank_values_doc'] > 1)} "
          f"hitdiv={sum(1 for r in sub if r['hit_div_doc_vs_ref'] > 0)}")
print("--- hit分歧行 ---")
for r in rows:
    if r["hit_div_doc_vs_ref"]:
        print(f"[{r['variant']}] {r['seed_q'][:40]}… doc={r['doc_ranks']}")
print("--- rank分歧行（hit同） ---")
for r in rows:
    if not r["hit_div_doc_vs_ref"] and r["n_rank_values_doc"] > 1:
        print(f"[{r['variant']}] {r['seed_q'][:40]}… doc={r['doc_ranks']}")

p = os.path.join(HERE, "w3_longq_results.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump({"n_probes": len(rows), "n_pure_flip": n_flip,
               "n_rankdiv": n_rankdiv, "n_hitdiv": n_hitdiv,
               "rows": rows}, f, ensure_ascii=False, indent=2)
print(f"[P3B] 结果已落盘 {p}")
