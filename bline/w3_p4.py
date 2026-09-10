#!/usr/bin/env python3
"""W3-P4：7格点 × v3-44全集（extended 35 chunk）+ 剔除敏感性分析。
只读基线；PYTHONDONTWRITEBYTECODE=1 运行。
用法：PYTHONDONTWRITEBYTECODE=1 python3 rag_min_demo/bline/w3_p4.py
"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from w3_harness import (EXPECT_EXTENDED_N, build_extended, eval_strategy,
                        git_sha, mcnemar_exact, strategies)

EXCLUDE_Q = {"纯向量检索在什么情况下会翻车", "长短文本的分数为什么不可比",
             "golden集应该包含什么"}

with open(os.path.join(HERE, "w1_config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
with open(os.path.join(HERE, "w3_golden_v3.json"), encoding="utf-8") as f:
    v3 = json.load(f)
assert v3["version"] == "w3-v3-44", v3["version"]
queries = v3["queries"]
assert len(queries) == 44, len(queries)
adv = [q for q in queries if "trap" in q]
seed = [q for q in queries if "trap" not in q]
assert len(adv) == 12 and len(seed) == 32, (len(seed), len(adv))
sens = [q for q in queries if q["q"] not in EXCLUDE_Q]
assert len(sens) == 41, len(sens)

chunks, index = build_extended()
assert len(chunks) == EXPECT_EXTENDED_N, len(chunks)
sts, names = strategies(cfg)
assert len(sts) == 7, len(sts)

def run_cell(st, name, qs):
    r = eval_strategy(st, chunks, index, qs, cfg["top_k"])
    r["strategy"] = name
    r["n_queries"] = len(qs)
    return r

cells, adv_cells, sens_cells = [], [], []
for st, name in zip(sts, names):
    c = run_cell(st, name, queries)
    a = run_cell(st, name, adv)
    s = run_cell(st, name, sens)
    cells.append({k: v for k, v in c.items() if k != "ranks"})
    adv_cells.append({k: v for k, v in a.items() if k != "ranks"})
    sens_cells.append({k: v for k, v in s.items() if k != "ranks"})
    print(f"  {name:12s} full44 doc={c['recall_at_k_doc']} MRR={c['mrr_doc']:.4f} "
          f"chunk={c['recall_at_k_chunk']} MRR={c['mrr_chunk']:.4f} | "
          f"adv12 doc={a['recall_at_k_doc']} | sens41 doc={s['recall_at_k_doc']}")

ref = next(c for c in cells if c["strategy"] == "w_cos=0.6")
ref_full_ranks = eval_strategy(sts[names.index("w_cos=0.6")], chunks, index,
                               queries, cfg["top_k"])["ranks"]
comp = []
for c, st in zip(cells, sts):
    rr = eval_strategy(st, chunks, index, queries, cfg["top_k"])["ranks"]
    b = sum(1 for a, x in zip(ref_full_ranks, rr)
            if a["doc_rank"] is None and x["doc_rank"] is not None)
    cc = sum(1 for a, x in zip(ref_full_ranks, rr)
             if a["doc_rank"] is not None and x["doc_rank"] is None)
    drank = sum(1 for a, x in zip(ref_full_ranks, rr)
                if a["doc_rank"] != x["doc_rank"])
    crank = sum(1 for a, x in zip(ref_full_ranks, rr)
                if a["chunk_rank"] != x["chunk_rank"])
    rh = [1 if r["chunk_rank"] else 0 for r in ref_full_ranks]
    xh = [1 if r["chunk_rank"] else 0 for r in rr]
    bb = sum(1 for a, x in zip(rh, xh) if a == 0 and x == 1)
    ccc = sum(1 for a, x in zip(rh, xh) if a == 1 and x == 0)
    comp.append({"strategy": c["strategy"], "b_doc": b, "c_doc": cc,
                 "disagree_hit_doc": b + cc,
                 "mcnemar_p_doc_vs_ref": mcnemar_exact(b, cc),
                 "disagree_rank_doc": drank,
                 "disagree_hit_chunk": bb + ccc,
                 "mcnemar_p_chunk_vs_ref": mcnemar_exact(bb, ccc),
                 "disagree_rank_chunk": crank})
    print(f"    vs ref {c['strategy']:12s} hit分歧doc={b+cc} "
          f"p={mcnemar_exact(b,cc)} rank分歧doc={drank} chunk={crank}")

out = {"generated_at": datetime.datetime.now().astimezone().isoformat(),
       "git_sha": git_sha(), "top_k": cfg["top_k"],
       "golden_version": v3["version"], "n_chunks": len(chunks),
       "cells_full44": cells, "cells_adv12": adv_cells,
       "cells_sens41_excl3drift": sens_cells,
       "excluded": sorted(EXCLUDE_Q),
       "vs_ref_w_cos06": comp,
       "honesty": ("P3对抗集7格点零分歧；P4只报告全集实测与剔除敏感性，"
                   "无信号则诚实判无信号，不做生产权重推荐。")}
p = os.path.join(HERE, "w3_results.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"[W3-P4] 全网格结果已落盘 {p}")
