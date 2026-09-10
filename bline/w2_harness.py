#!/usr/bin/env python3
"""W2评测harness：32题SILVER集上的权重网格对照。

只读复用冻结基线（import rag_demo，不改其文件）与W1纯函数（w1_hybrid）。
统计：recall@3/MRR + Wilson 95% CI + 策略间分歧计数 + 相对基线等价策略
(w_cos=0.6)的配对McNemar exact双侧p（stdlib math.comb实现，无新依赖）。
确定性：无随机成分。用法：PYTHONDONTWRITEBYTECODE=1 python3 rag_min_demo/bline/w2_harness.py
"""
import datetime
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # 冻结基线 rag_demo（只读）
import rag_demo
from w1_hybrid import rrf_fusion, weighted_fusion

Z = 1.96


def wilson(k, n):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1.0 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def mcnemar_exact(b, c):
    """b=ref错新对，c=ref对新错；双侧exact p。"""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    tail = sum(math.comb(n, k) for k in range(m + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def load_inputs():
    with open(os.path.join(HERE, "w1_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    with open(os.path.join(HERE, "w2_golden.json"), encoding="utf-8") as f:
        golden = json.load(f)
    return cfg, golden


def score_for(strategy, chunks, index, query, top_k):
    qtoks = rag_demo.bigrams(query)
    qvec = index._vec(qtoks)
    if not any(qvec.values()):
        return []
    cos = {i: s for s, i in index.cosine(qvec)}
    bm = {i: s for s, i in index.bm25(qtoks)}
    if strategy[0] == "weighted":
        ranked = weighted_fusion(cos, bm, w_cos=strategy[1],
                                 candidates=strategy[2], top_k=top_k)
    else:
        ranked = rrf_fusion(cos, bm, k=strategy[1],
                            candidates=strategy[2], top_k=top_k)
    return [(s, chunks[i]) for s, i in ranked]


def eval_strategy(strategy, chunks, index, queries, top_k):
    ranks, hits, rr = [], 0, 0.0
    for item in queries:
        res = score_for(strategy, chunks, index, item["q"], top_k)
        rank = next((i + 1 for i, (_, c) in enumerate(res)
                     if c["doc_id"] == item["expect"]), None)
        top1 = res[0][1]["doc_id"] if res else None
        ranks.append({"q": item["q"], "expect": item["expect"],
                      "type": item.get("type"), "rank": rank, "top1": top1})
        hits += rank is not None
        rr += 1.0 / rank if rank else 0.0
    n = len(queries)
    return {"recall_at_k": [hits, n], "mrr": round(rr / n, 4),
            "wilson95_recall": wilson(hits, n), "ranks": ranks,
            "hit_vec": [1 if r["rank"] else 0 for r in ranks]}


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.getcwd(), text=True).strip()
    except Exception:
        return "unknown"


def main():
    cfg, golden = load_inputs()
    queries = golden["queries"]
    chunks, index = rag_demo.run_pipeline(verbose=False)
    strategies = [("weighted", w, cfg["candidates"]) for w in cfg["w_cos_grid"]]
    strategies.append(("rrf", cfg["rrf_k"], cfg["candidates"]))
    names = [f"w_cos={s[1]}" if s[0] == "weighted" else f"rrf_k={s[1]}"
             for s in strategies]
    cells = []
    for st, name in zip(strategies, names):
        r = eval_strategy(st, chunks, index, queries, cfg["top_k"])
        r["strategy"] = name
        # W1连续性子集：5道orig题
        sub = [x for x in r["ranks"] if x["type"] == "orig"]
        r["w1_subset"] = {"recall_at_k": [sum(1 for x in sub if x["rank"]), len(sub)],
                          "mrr": round(sum(1.0 / x["rank"] for x in sub if x["rank"]) / len(sub), 4)}
        cells.append(r)
        print(f"  {name:12s} recall@{cfg['top_k']}={r['recall_at_k'][0]}/{r['recall_at_k'][1]}"
              f"  MRR={r['mrr']:.3f}  Wilson95={r['wilson95_recall']}"
              f"  W1子集={r['w1_subset']['recall_at_k'][0]}/{r['w1_subset']['recall_at_k'][1]}")
    # 基线等价策略(w_cos=0.6)为参照：分歧计数 + McNemar
    ref = next(c for c in cells if c["strategy"] == "w_cos=0.6")
    comp = []
    for c in cells:
        b = sum(1 for a, x in zip(ref["hit_vec"], c["hit_vec"]) if a == 0 and x == 1)
        cc = sum(1 for a, x in zip(ref["hit_vec"], c["hit_vec"]) if a == 1 and x == 0)
        disagree_rank = sum(1 for a, x in zip(ref["ranks"], c["ranks"]) if a["rank"] != x["rank"])
        comp.append({"strategy": c["strategy"],
                     "disagree_hit": b + cc, "b": b, "c": cc,
                     "mcnemar_p_vs_ref": mcnemar_exact(b, cc),
                     "disagree_rank": disagree_rank})
        print(f"    vs ref {c['strategy']:12s} 分歧hit={b + cc} (b={b},c={cc})"
              f" p={comp[-1]['mcnemar_p_vs_ref']} 分歧rank={disagree_rank}")
    # 按期望文档拆分（参照策略）
    by_doc = {}
    for r in ref["ranks"]:
        d = by_doc.setdefault(r["expect"], {"n": 0, "hits": 0, "miss_q": []})
        d["n"] += 1
        if r["rank"]:
            d["hits"] += 1
        else:
            d["miss_q"].append(r["q"])
    print("  按文档（ref策略）:", {k: f"{v['hits']}/{v['n']}" for k, v in by_doc.items()})
    out = {"generated_at": datetime.datetime.now().astimezone().isoformat(),
           "git_sha": git_sha(), "n_queries": len(queries),
           "baseline_frozen": {"recall_at_3": [5, 5], "mrr": 1.0,
                               "source": "rag_demo.py EVAL_SET（W2复跑仍5/5，见报告头）"},
           "config": cfg, "cells": cells,
           "vs_ref_w_cos06": comp, "by_doc_ref": by_doc,
           "honesty": "SILVER n=32仍是小规模玩具语料；McNemar在零分歧时p=1无信息量，"
                      "结论限于本语料，不做生产权重推荐。"}
    p = os.path.join(HERE, "w2_results.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[W2] 全网格 {len(cells)} 格点已落盘 {p}")


if __name__ == "__main__":
    sys.exit(main())
