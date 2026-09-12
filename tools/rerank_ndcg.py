#!/usr/bin/env python3
"""B 线 rerank + nDCG 度量（RankGPT 思想的非 LLM 版）。

现状：B 线只有 recall@3 / MRR（二元命中，无排序质量梯度）。本模块补：
  1) nDCG@k 度量（支持分级相关度；本集用二元 gain，等价于对排序位置的折扣求和）
  2) 一个确定性 reranker（查询二元组覆盖率），与冻结基线融合后重排
  3) 在 W2(32) 上对照实验：rerank 是否真的提升排序质量

诚实边界（P2 B-P38 RankGPT 是**列式 LLM rerank**）：本仓无 LLM、无权重，
  故只落地 nDCG 度量 + 非 LLM 确定性 reranker；RankGPT 的 prompt 模板与
  LLM 调用**未实现**（记为未迁移，不做装饰性占位）。

只读：不修改 rag_demo.py / docs/（FINAL_ACCEPTANCE 冻结）。

运行：python3 tools/rerank_ndcg.py
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import rag_demo as R  # noqa: E402

W2_GOLDEN = os.path.join(ROOT, "bline", "w2_golden.json")
ALPHA = 0.5  # 基线分权重；1-ALPHA 给覆盖率


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _dedupe_docs(seq):
    """文档级去重（保序）。

    缺陷修正：chunk 级检索会让**同一文档的多个 chunk** 占据多个名次，
    若直接按 chunk 计 nDCG，同一相关文档被重复计分 → nDCG 可 > 1
    （实测 1.63）。文档级去重后再算，才是合法 nDCG。
    """
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def ndcg_at_k(ranked_ids, relevant, k=3):
    """nDCG@k（文档级去重后计算，保证 ∈ [0,1]）。

    relevant 可为 dict(id->gain) 或 ids 的可迭代。
    """
    ranked_ids = _dedupe_docs(ranked_ids)
    if isinstance(relevant, dict):
        rel = relevant
    else:
        rel = {r: 1.0 for r in relevant}
    gains = [rel.get(d, 0.0) for d in ranked_ids[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    val = 0.0 if idcg == 0 else dcg(gains) / idcg
    # 不变量守卫：文档级去重后 nDCG 必须 ∈ [0,1]，否则度量实现有缺陷
    if not (-1e-9 <= val <= 1 + 1e-9):
        raise AssertionError(
            f"nDCG 越界 {val}（ids={ranked_ids[:k]}, rel={rel}）——度量实现缺陷"
        )
    return val


def coverage(toks, qtoks):
    """查询二元组覆盖率：查询里有多少二元组出现在该 chunk（召回取向信号）。"""
    if not qtoks:
        return 0.0
    present = sum(1 for t in set(qtoks) if t in set(toks))
    return present / len(set(qtoks))


def ranked_with_coverage(query, chunks, index, top_k=3, alpha=ALPHA):
    """基线余弦分 + 覆盖率 融合重排（确定性，无 LLM）。"""
    qtoks = R.bigrams(query)
    qvec = index._vec(qtoks)
    if not any(qvec.values()):
        return []
    cos = {i: s for s, i in index.cosine(qvec)}
    cand = sorted(cos, key=cos.get, reverse=True)[:R.CANDIDATES]
    cmax = max(cos[i] for i in cand) or 1.0
    scored = []
    for i in cand:
        base = cos[i] / cmax
        cov = coverage(R.bigrams(chunks[i]["text"]), qtoks)
        scored.append((alpha * base + (1 - alpha) * cov, chunks[i]["doc_id"]))
    scored.sort(reverse=True)
    return [(s, d) for s, d in scored[:top_k]]


def ranked_baseline(query, chunks, index, top_k=3):
    """冻结基线检索（rag_demo.retrieve）。"""
    return [(s, c["doc_id"]) for s, c in R.retrieve(query, chunks, index, top_k=top_k)]


def ranked_diverse(query, chunks, index, top_k=3):
    """文档多样重排：同一文档只保留最高分 chunk，再取名次。

    动因（实测发现）：冻结基线有 18/32 查询的 top-3 内含同文档多 chunk，
    会把正确文档挤出 top-3（唯一的 miss 即由此而来）。此重排针对该缺陷。
    """
    qtoks = R.bigrams(query)
    qvec = index._vec(qtoks)
    if not any(qvec.values()):
        return []
    cos = {i: s for s, i in index.cosine(qvec)}
    cand = sorted(cos, key=cos.get, reverse=True)[:max(R.CANDIDATES, top_k * 4)]
    best = {}
    for i in cand:
        d = chunks[i]["doc_id"]
        if d not in best or cos[i] > best[d][0]:
            best[d] = (cos[i], d)
    return [(s, d) for s, d in sorted(best.values(), reverse=True)[:top_k]]


def evaluate(rows, name):
    n = len(rows)
    dup = sum(1 for r in rows if len(set(r["ids"])) < len(r["ids"]))
    recall = sum(1 for r in rows if r["expect"] in r["ids"]) / n
    mrr = sum((1.0 / (r["ids"].index(r["expect"]) + 1)
               if r["expect"] in r["ids"] else 0.0) for r in rows) / n
    ndcg = sum(ndcg_at_k(r["ids"], {r["expect"]}, k=3) for r in rows) / n
    print(f"  {name:22s} recall@3={recall:.3f}  MRR={mrr:.3f}  nDCG@3={ndcg:.3f}"
          f"  (含重复文档的名次 {dup}/{n})")
    return {"recall": round(recall, 4), "mrr": round(mrr, 4),
            "ndcg": round(ndcg, 4), "dup": dup}


def main():
    with open(W2_GOLDEN, encoding="utf-8") as f:
        qa = [(it["q"], it["expect"]) for it in json.load(f)["queries"]]
    chunks, index = R.run_pipeline(verbose=False)

    arms = {
        "冻结基线 余弦+BM25": ranked_baseline,
        f"rerank 余弦+覆盖率(a={ALPHA})": lambda q, c, i: ranked_with_coverage(q, c, i),
        "rerank 文档多样": ranked_diverse,
    }
    results = {}
    print(f"[W2(32) 三臂对照]")
    for name, fn in arms.items():
        rows = [{"expect": exp, "ids": [d for _, d in fn(q, chunks, index)]}
                for q, exp in qa]
        results[name] = evaluate(rows, name)

    base = results["冻结基线 余弦+BM25"]
    print()
    best_name, best = None, None
    for name, r in results.items():
        if name == "冻结基线 余弦+BM25":
            continue
        d_ndcg = r["ndcg"] - base["ndcg"]
        d_rec = r["recall"] - base["recall"]
        print(f"  {name:26s} ΔnDCG {d_ndcg:+.4f}  Δrecall {d_rec:+.4f}")
        if best is None or (r["ndcg"], r["recall"]) > (best["ndcg"], best["recall"]):
            best_name, best = name, r

    verdict = ("文档多样重排有增益" if best["ndcg"] > base["ndcg"]
               else "重排无增益（如实记录：本集上两种信号均未超过冻结基线）")
    print(f"  → {verdict}")

    # 非空性：倒序名次，nDCG 必须下降（否则度量恒真）
    rows_base = [{"expect": exp, "ids": [d for _, d in ranked_baseline(q, chunks, index)]}
                 for q, exp in qa]
    shuffled = sum(ndcg_at_k(list(reversed(rr["ids"])), {rr["expect"]}, k=3)
                   for rr in rows_base) / len(rows_base)
    print(f"  [非空性] 倒序名次 nDCG@3 = {shuffled:.3f} "
          f"({'< 正序 ✓' if shuffled < base['ndcg'] else '异常：度量恒真 ✗'})")
    if shuffled >= base["ndcg"]:
        print("RERANK GATE FAIL: nDCG 度量非空性失守")
        return 1

    # 不变量：所有 nDCG 必须 ∈ [0,1]（去重修正的守卫）
    for name, r in results.items():
        if not (0.0 <= r["ndcg"] <= 1.0):
            print(f"RERANK GATE FAIL: {name} nDCG={r['ndcg']} 越界")
            return 1

    print("RERANK/nDCG GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
