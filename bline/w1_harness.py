#!/usr/bin/env python3
"""W1评测harness：只读复用冻结基线（import rag_demo，不改其文件），跑权重网格对照。

输出 bline/w1_results.json：全部格点 + 逐题rank，全量报告不挑选。
确定性：无随机成分，同代码同语料下结果字节可复现。
用法：python3 rag_min_demo/bline/w1_harness.py
"""
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # 导入冻结基线 rag_demo（只读）
import rag_demo
from w1_hybrid import rrf_fusion, weighted_fusion


def load_inputs():
    with open(os.path.join(HERE, "w1_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    with open(os.path.join(HERE, "w1_golden.json"), encoding="utf-8") as f:
        golden = json.load(f)
    return cfg, golden


def build_index():
    chunks, index = rag_demo.run_pipeline(verbose=False)
    return chunks, index


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


def eval_strategy(strategy, chunks, index, golden, top_k):
    ranks, hits, rr = [], 0, 0.0
    for item in golden["queries"]:
        res = score_for(strategy, chunks, index, item["q"], top_k)
        rank = next((i + 1 for i, (_, c) in enumerate(res)
                     if c["doc_id"] == item["expect"]), None)
        ranks.append({"q": item["q"], "expect": item["expect"], "rank": rank})
        hits += rank is not None
        rr += 1.0 / rank if rank else 0.0
    n = len(golden["queries"])
    return {"recall_at_k": [hits, n], "mrr": round(rr / n, 4), "ranks": ranks}


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.getcwd(), text=True).strip()
    except Exception:
        return "unknown"


def main():
    cfg, golden = load_inputs()
    chunks, index = build_index()
    strategies = [("weighted", w, cfg["candidates"]) for w in cfg["w_cos_grid"]]
    strategies.append(("rrf", cfg["rrf_k"], cfg["candidates"]))
    cells = []
    for st in strategies:
        name = f"w_cos={st[1]}" if st[0] == "weighted" else f"rrf_k={st[1]}"
        r = eval_strategy(st, chunks, index, golden, cfg["top_k"])
        r["strategy"] = name
        cells.append(r)
        print(f"  {name:12s} recall@{cfg['top_k']}={r['recall_at_k'][0]}/{r['recall_at_k'][1]}"
              f"  MRR={r['mrr']:.3f}")
    out = {"generated_at": datetime.datetime.now().astimezone().isoformat(),
           "git_sha": git_sha(),
           "baseline_frozen": {"recall_at_3": [5, 5], "mrr": 1.0,
                               "source": "rag_demo.py EVAL_SET（本次复跑已复现）"},
           "config": cfg, "cells": cells,
           "honesty": "n=5为冒烟规模；网格内任何权重差异均可能是过拟合，不做生产权重推荐。"}
    p = os.path.join(HERE, "w1_results.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[W1] 全网格 {len(cells)} 格点已落盘 {p}")


if __name__ == "__main__":
    sys.exit(main())
