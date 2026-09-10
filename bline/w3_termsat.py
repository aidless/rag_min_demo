#!/usr/bin/env python3
"""W3-P3B term_saturation挖掘：关键词堆叠/罕见词重复是否使7格点分歧。

机制假设（见rag_demo.py实测代码，非猜测）：
- cosine路：query向量TF用(1+log(tf))加权后L2归一化，重复同一词只按log增长，
  且会稀释查询中其他词的权重份额；
- BM25路：`qtf = Counter(qtoks)`只用于按类型迭代（`for t, qc in qtf.items()`），
  qc从不乘入分数——query词频对BM25完全不可见，重复堆叠不改变BM25任一分数。
所以“罕见词重复”是纯cos路扰动（非对称探针，此处如实披露）；BM25只在改写
引入新词类型时才动。两路共享cos top-candidates池（candidates=10），分歧只能
来自池内重排。
全部用真实语料统计构造，不编造文本：
  T-rep3   q + 金chunk最罕见二元组×3（按index.idf排名）
  T-rep8   q + 金chunk最罕见二元组×8
  T-stack  q + 金chunk top5罕见二元组拼接（关键词堆叠）
注：同义改写需人工写自然问法（待立项），本脚本的机械堆叠是其下界替代。
口径：7格点（沿用w1_config），top_k=3；只记录分歧，不污染v3 golden。
这些是机制探针（含查询端词频操纵），结论只用于“权重是否有信号”。
用法：PYTHONDONTWRITEBYTECODE=1 python3 rag_min_demo/bline/w3_termsat.py
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


def rarest_bigrams(gold_text, k=5):
    toks = rag_demo.bigrams(gold_text)
    uniq = sorted(set(toks), key=lambda t: (-index.idf.get(t, 0.0), t))
    return uniq[:k]


def pure_tops(query, k=3):
    qtoks = rag_demo.bigrams(query)
    qvec = index._vec(qtoks)
    cos = sorted(index.cosine(qvec), reverse=True)[:k]
    bm = sorted(index.bm25(qtoks), reverse=True)[:k]
    return ([chunks[i]["chunk_id"] for _, i in cos],
            [chunks[i]["chunk_id"] for _, i in bm])


def run_variant(qtext, doc, chs):
    rec = {"doc_rank": {}, "chunk_rank": {}}
    for st, name in zip(sts, names):
        res = score_for(st, chunks, index, qtext, cfg["top_k"])
        d = next((i + 1 for i, (_, c) in enumerate(res)
                  if c["doc_id"] == doc), None)
        kk = next((i + 1 for i, (_, c) in enumerate(res)
                   if chs and c["chunk_id"] in chs), None)
        rec["doc_rank"][name] = d
        rec["chunk_rank"][name] = kk
    return rec


rows, n_nogold = [], 0
for item in seeds:
    doc, chs = norm_item(item)
    if not chs:
        n_nogold += 1
        continue
    gold_text = by_id[chs[0]][1]["text"]
    rare = rarest_bigrams(gold_text, 5)
    R, STACK = rare[0], "".join(rare)
    variants = {"T-rep3": item["q"] + "\n" + R * 3,
                "T-rep8": item["q"] + "\n" + R * 8,
                "T-stack": item["q"] + "\n" + STACK}
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
                     "rare_bigram": R, "stack": STACK, "qlen": len(qtext),
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
print(f"[P3B-termsat] 探针={len(rows)} 无金跳过seed={n_nogold} "
      f"纯路top1翻转={n_flip} 7格点rank分歧={n_rankdiv} hit分歧={n_hitdiv}")
for v in ("T-rep3", "T-rep8", "T-stack"):
    sub = [r for r in rows if r["variant"] == v]
    print(f"  {v:8s} n={len(sub)} flip={sum(1 for r in sub if r['pure_flip'])} "
          f"rankdiv={sum(1 for r in sub if r['n_rank_values_doc'] > 1)} "
          f"hitdiv={sum(1 for r in sub if r['hit_div_doc_vs_ref'] > 0)}")
print("--- hit分歧行 ---")
for r in rows:
    if r["hit_div_doc_vs_ref"]:
        print(f"[{r['variant']}] {r['seed_q'][:40]}… rare={r['rare_bigram']} "
              f"doc={r['doc_ranks']}")
print("--- rank分歧行（hit同） ---")
for r in rows:
    if not r["hit_div_doc_vs_ref"] and r["n_rank_values_doc"] > 1:
        print(f"[{r['variant']}] {r['seed_q'][:40]}… rare={r['rare_bigram']} "
              f"doc={r['doc_ranks']}")

p = os.path.join(HERE, "w3_termsat_results.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump({"n_probes": len(rows), "n_nogold_skipped": n_nogold,
               "n_pure_flip": n_flip, "n_rankdiv": n_rankdiv,
               "n_hitdiv": n_hitdiv, "rows": rows},
              f, ensure_ascii=False, indent=2)
print(f"[P3B-termsat] 结果已落盘 {p}")
