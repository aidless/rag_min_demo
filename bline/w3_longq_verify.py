#!/usr/bin/env python3
"""P3B复核：对首个hit分歧用例做独立重算 + 填充长度敏感性检查。

复核点：
1. 同一长查询重跑7格点，结果与w3_longq_results.json逐格一致（确定性）。
2. 纯cos/BM25两路分数明细：谁赢、margin多少（机制归因）。
3. 填充截断扫描（near填充取前50/100/200字）：分歧是否随填充长度单调出现
   （可控信号）还是孤立抖动（fluke）。
只读基线与已有文件；输出复核JSON。
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
seeds = [q for q in v3["queries"] if "trap" not in q]
with open(os.path.join(HERE, "w3_longq_results.json"), encoding="utf-8") as f:
    prev = json.load(f)

HIT = next(r for r in prev["rows"] if r["hit_div_doc_vs_ref"] > 0)
print(f"复核对象: [{HIT['variant']}] {HIT['seed_q'][:50]}…")
item = next(q for q in seeds if q["q"] == HIT["seed_q"])
doc, chs = norm_item(item)
qlen_short = len(item["q"])
filler = HIT["qlen"] - qlen_short - 1
print(f"  短问len={qlen_short} 填充len≈{filler} expect={doc}/{chs}")

chunks, index = build_extended()
sts, names = strategies(cfg)

# 1. 确定性重算
# 从构造规则重构：L-fill-near = q + ref短问首中非金chunk原文
ref_st = sts[names.index("w_cos=0.6")]
res0 = score_for(ref_st, chunks, index, item["q"], cfg["top_k"])
near_text = next(c["text"] for _, c in res0 if c["doc_id"] != doc)
qtext = item["q"] + "\n" + near_text
assert len(qtext) == HIT["qlen"], (len(qtext), HIT["qlen"])
rerun = {}
for st, name in zip(sts, names):
    res = score_for(st, chunks, index, qtext, cfg["top_k"])
    d = next((i + 1 for i, (_, c) in enumerate(res)
              if c["doc_id"] == doc), None)
    rerun[name] = d
assert rerun == HIT["doc_ranks"], (rerun, HIT["doc_ranks"])
print("  [PASS] 7格点逐格重算一致（确定性）")

# 2. 纯路分数明细
qtoks = rag_demo.bigrams(qtext)
qvec = index._vec(qtoks)
cos = {i: s for s, i in index.cosine(qvec)}
bm = {i: s for s, i in index.bm25(qtoks)}
top_c = sorted(cos, key=cos.get, reverse=True)[:5]
top_b = sorted(bm, key=bm.get, reverse=True)[:5]
print("  纯cos top5:", [(chunks[i]["chunk_id"], round(cos[i], 4))
                        for i in top_c])
print("  纯BM25 top5:", [(chunks[i]["chunk_id"], round(bm[i], 4))
                         for i in top_b])
gold_idx = [i for i, c in enumerate(chunks) if c["chunk_id"] in chs]
print(f"  金chunk纯cos排名: {sorted(cos, key=cos.get, reverse=True).index(gold_idx[0]) + 1}/35, "
      f"纯BM25排名: {sorted(bm, key=bm.get, reverse=True).index(gold_idx[0]) + 1}/35, "
      f"cos候选池内: {gold_idx[0] in top_c or gold_idx[0] in sorted(cos, key=cos.get, reverse=True)[:10]}")
for i in top_c[:3]:
    print(f"    cos {chunks[i]['chunk_id']}: cos={cos[i]:.4f} bm25={bm[i]:.4f}")
for i in top_b[:3]:
    if i not in top_c[:3]:
        print(f"    bm25 {chunks[i]['chunk_id']}: cos={cos[i]:.4f} bm25={bm[i]:.4f}")

# 3. 填充长度扫描
print("  填充截断扫描（near填充前N字）:")
scan = {}
for n in (50, 100, 200, 10 ** 9):
    qt = item["q"] + "\n" + near_text[:n]
    rr = {}
    for st, name in zip(sts, names):
        res = score_for(st, chunks, index, qt, cfg["top_k"])
        rr[name] = next((i + 1 for i, (_, c) in enumerate(res)
                         if c["doc_id"] == doc), None)
    vals = set(str(v) for v in rr.values())
    scan[n] = rr
    print(f"    fill≤{n if n < 10**9 else '全文'}字: {rr} 分歧值集={sorted(vals)}")

out = {"seed_q": HIT["seed_q"], "variant": HIT["variant"],
       "deterministic_rerun_match": True,
       "pure_cos_top5": [(chunks[i]["chunk_id"], cos[i]) for i in top_c],
       "pure_bm25_top5": [(chunks[i]["chunk_id"], bm[i]) for i in top_b],
       "fill_scan": {str(k): v for k, v in scan.items()}}
p = os.path.join(HERE, "w3_longq_verify.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"[P3B复核] 已落盘 {p}")
