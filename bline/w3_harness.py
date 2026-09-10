#!/usr/bin/env python3
"""W3-P2 chunk级评测harness：doc级+chunk级双口径 + W2回放门禁。

只读复用冻结基线（import rag_demo 的 parse_docs/chunk_text/bigrams/
TfidfIndex，不改其文件）与W1纯函数（w1_hybrid）。两种管线：
  replay   = 冻结 run_pipeline() 原样（docs/ 10 chunk），W2集回放，
             逐格点断言与 w2_results.json 一致（recall+MRR），不一致即非零退出。
  extended = docs/ + corpus/ 35 chunk（id稳定：doc_id#cN），跑w3 golden双口径。
统计：recall@k/MRR/Wilson95 + 相对ref(w_cos=0.6)分歧与McNemar exact。
确定性：无随机成分。
用法：PYTHONDONTWRITEBYTECODE=1 python3 rag_min_demo/bline/w3_harness.py
"""
import datetime
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                              # w1_hybrid / w2 结果
sys.path.insert(0, os.path.dirname(HERE))             # 冻结基线 rag_demo（只读）
import rag_demo
from w1_hybrid import rrf_fusion, weighted_fusion

Z = 1.96
EXPECT_EXTENDED_N = 35  # SNAPSHOT快照总数（P1实测）
# P1勘误：SNAPSHOT.md落盘后，count_chunks.py复算会把它自身也切进去
# （35+10=45）。SNAPSHOT是评测基建不是被检索语料，此处按名排除，
# 与P1的35口径对齐；P1文件保持字节冻结，勘误记入PHASE_LOG。
SKIP_DOCS = {"SNAPSHOT"}


def wilson(k, n):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1.0 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    tail = sum(math.comb(n, k) for k in range(m + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def build_replay():
    """冻结管线原样（与W2同一调用）。"""
    return rag_demo.run_pipeline(verbose=False)


def build_extended():
    """docs/+corpus/双源切分，逻辑逐行对应 run_pipeline（只读复用函数）。"""
    root = os.path.dirname(HERE)
    chunks = []
    for d in (os.path.join(root, "docs"), os.path.join(HERE, "corpus")):
        for doc in rag_demo.parse_docs(d):
            if doc["doc_id"] in SKIP_DOCS:
                continue  # 评测基建文件，不进检索池（见P1勘误）
            for j, c in enumerate(rag_demo.chunk_text(doc["text"])):
                chunks.append({"chunk_id": f"{doc['doc_id']}#c{j}",
                               "doc_id": doc["doc_id"], "title": doc["title"],
                               "text": c, "chars": len(c)})
    toks = [rag_demo.bigrams(c["text"]) for c in chunks]
    rag_demo.chunk_tokens_global = toks  # 与 run_pipeline 同步全局（BM25复用）
    index = rag_demo.TfidfIndex(toks)
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


def norm_item(item):
    """兼容W2单标注（expect）与W3双标注（expect_doc/expect_chunks）。"""
    doc = item.get("expect_doc", item.get("expect"))
    ch = item.get("expect_chunks")
    ch = list(ch) if ch else None
    return doc, ch


def eval_strategy(strategy, chunks, index, queries, top_k):
    ranks, dh, dr, ch, cr = [], 0, 0.0, 0, 0.0
    n_chunk_labeled = 0
    for item in queries:
        doc, chunk_ids = norm_item(item)
        res = score_for(strategy, chunks, index, item["q"], top_k)
        drank = next((i + 1 for i, (_, c) in enumerate(res)
                      if c["doc_id"] == doc), None)
        crank = (next((i + 1 for i, (_, c) in enumerate(res)
                       if c["chunk_id"] in chunk_ids), None)
                 if chunk_ids else None)
        ranks.append({"q": item["q"], "expect_doc": doc,
                      "expect_chunks": chunk_ids,
                      "doc_rank": drank, "chunk_rank": crank})
        dh += drank is not None
        dr += 1.0 / drank if drank else 0.0
        if chunk_ids is not None:
            n_chunk_labeled += 1
            ch += crank is not None
            cr += 1.0 / crank if crank else 0.0
    n = len(queries)
    out = {"recall_at_k_doc": [dh, n], "mrr_doc": round(dr / n, 4),
           "wilson95_doc": wilson(dh, n),
           "hit_vec_doc": [1 if r["doc_rank"] else 0 for r in ranks]}
    if n_chunk_labeled:
        out.update({"n_chunk_labeled": n_chunk_labeled,
                    "recall_at_k_chunk": [ch, n_chunk_labeled],
                    "mrr_chunk": round(cr / n_chunk_labeled, 4),
                    "wilson95_chunk": wilson(ch, n_chunk_labeled)})
    out["ranks"] = ranks
    return out


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.getcwd(), text=True).strip()
    except Exception:
        return "unknown"


def strategies(cfg):
    sts = [("weighted", w, cfg["candidates"]) for w in cfg["w_cos_grid"]]
    sts.append(("rrf", cfg["rrf_k"], cfg["candidates"]))
    names = [f"w_cos={s[1]}" if s[0] == "weighted" else f"rrf_k={s[1]}"
             for s in sts]
    return sts, names


def main():
    with open(os.path.join(HERE, "w1_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    with open(os.path.join(HERE, "w2_golden.json"), encoding="utf-8") as f:
        w2 = json.load(f)
    with open(os.path.join(HERE, "w2_results.json"), encoding="utf-8") as f:
        w2ref = json.load(f)
    with open(os.path.join(HERE, "w3_golden.json"), encoding="utf-8") as f:
        w3 = json.load(f)
    sts, names = strategies(cfg)
    ref_cells = {c["strategy"]: c for c in w2ref["cells"]}

    # ---- 回放门禁：docs-only冻结管线 + W2集，逐格点对账 ----
    print("[replay] docs-only冻结管线 × W2-32题 …")
    chunks_r, index_r = build_replay()
    assert len(chunks_r) == 10, f"冻结管线chunk数漂移：{len(chunks_r)}"
    replay_ok, replay = True, []
    for st, name in zip(sts, names):
        r = eval_strategy(st, chunks_r, index_r, w2["queries"], cfg["top_k"])
        exp = ref_cells[name]
        ok = (r["recall_at_k_doc"] == exp["recall_at_k"]
              and r["mrr_doc"] == exp["mrr"])
        replay_ok &= ok
        replay.append({"strategy": name, "recall_at_k_doc": r["recall_at_k_doc"],
                       "mrr_doc": r["mrr_doc"],
                       "expect_recall": exp["recall_at_k"], "expect_mrr": exp["mrr"],
                       "match": ok})
        print(f"  {name:12s} recall={r['recall_at_k_doc']} MRR={r['mrr_doc']}"
              f"  期望{exp['recall_at_k']}/{exp['mrr']}  {'PASS' if ok else 'FAIL'}")
    print(f"[replay] 门禁：{'PASS' if replay_ok else 'FAIL'}")

    # ---- 扩展管线：35 chunk × W3 seed双口径 ----
    print("[extended] docs+corpus扩展管线 × W3-seed …")
    chunks_e, index_e = build_extended()
    assert len(chunks_e) == EXPECT_EXTENDED_N, f"扩展chunk数漂移：{len(chunks_e)}"
    ids = sorted(c["chunk_id"] for c in chunks_e)
    assert len(set(ids)) == len(ids), "chunk_id碰撞"
    cells = []
    for st, name in zip(sts, names):
        r = eval_strategy(st, chunks_e, index_e, w3["queries"], cfg["top_k"])
        r["strategy"] = name
        cells.append(r)
        cl = (f"chunk={r['recall_at_k_chunk'][0]}/{r['recall_at_k_chunk'][1]}"
              f" MRR={r['mrr_chunk']:.3f}" if "recall_at_k_chunk" in r else "chunk=未标注")
        print(f"  {name:12s} doc={r['recall_at_k_doc'][0]}/{r['recall_at_k_doc'][1]}"
              f" MRR={r['mrr_doc']:.3f}  {cl}")
    ref = next(c for c in cells if c["strategy"] == "w_cos=0.6")
    comp = []
    for c in cells:
        b = sum(1 for a, x in zip(ref["hit_vec_doc"], c["hit_vec_doc"])
                if a == 0 and x == 1)
        cc = sum(1 for a, x in zip(ref["hit_vec_doc"], c["hit_vec_doc"])
                 if a == 1 and x == 0)
        drank = sum(1 for a, x in zip(ref["ranks"], c["ranks"])
                    if a["doc_rank"] != x["doc_rank"])
        item = {"strategy": c["strategy"], "disagree_hit_doc": b + cc,
                "b": b, "c": cc, "mcnemar_p_vs_ref": mcnemar_exact(b, cc),
                "disagree_rank_doc": drank}
        if "recall_at_k_chunk" in ref and "recall_at_k_chunk" in c:
            rh = [1 if r["chunk_rank"] else 0 for r in ref["ranks"]]
            xh = [1 if r["chunk_rank"] else 0 for r in c["ranks"]]
            bb = sum(1 for a, x in zip(rh, xh) if a == 0 and x == 1)
            ccc = sum(1 for a, x in zip(rh, xh) if a == 1 and x == 0)
            item.update({"disagree_hit_chunk": bb + ccc,
                         "mcnemar_p_chunk_vs_ref": mcnemar_exact(bb, ccc)})
        comp.append(item)
        print(f"    vs ref {c['strategy']:12s} doc分歧hit={b + cc} p={item['mcnemar_p_vs_ref']}"
              + (f" chunk分歧hit={item['disagree_hit_chunk']} p={item['mcnemar_p_chunk_vs_ref']}"
                 if "disagree_hit_chunk" in item else ""))
    # 干扰信号：相对冻结管线，W2题doc命中的得失
    out = {"generated_at": datetime.datetime.now().astimezone().isoformat(),
           "git_sha": git_sha(), "top_k": cfg["top_k"],
           "replay_gate": {"pass": replay_ok, "detail": replay},
           "extended_n_chunks": len(chunks_e),
           "n_queries": len(w3["queries"]), "cells": cells,
           "vs_ref_w_cos06": comp,
           "honesty": "P2只落基建：扩展管线+双口径+回放门禁；权重优劣结论等P4全集实测。"}
    p = os.path.join(HERE, "w3_p2_check.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[W3-P2] 双口径结果已落盘 {p}")
    if not replay_ok:
        print("[W3-P2] 回放门禁FAIL，P2 gate不通过")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
