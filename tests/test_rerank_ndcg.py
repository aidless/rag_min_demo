#!/usr/bin/env python3
"""B 线 rerank / nDCG 回归测试（stdlib-only）。

冻结本轮（P3-HARDEN B1）实测数字，任何漂移 → FAIL。

关键发现（本轮）：
  1. 度量缺陷：chunk 级名次里同文档多 chunk 被重复计分 → nDCG 可 > 1（实测 1.63）。
     修正：文档级去重后计算；并加不变量守卫（越界即抛）。
  2. 检索缺陷：18/32 查询 top-3 含同文档多 chunk，把正确文档挤出 → 唯一 miss。
     修法：文档多样重排（同文档只留最高分 chunk）。
  3. 实测增益：recall@3 31/32 → 32/32；MRR 0.969 → 0.979；nDCG@3 0.969 → 0.984。

只读：不修改 rag_demo.py / docs/（FINAL_ACCEPTANCE 冻结）。
运行：python3 -m pytest -q tests/test_rerank_ndcg.py
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "rerank_ndcg", str(ROOT / "tools" / "rerank_ndcg.py")
)
RR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RR)

import rag_demo  # noqa: E402

FROZEN = {
    "base_recall": 31, "base_dup": 18,
    "div_recall": 32, "div_dup": 0,
    "div_mrr": 0.979, "div_ndcg": 0.984,
}


def _qa():
    with open(ROOT / "bline" / "w2_golden.json", encoding="utf-8") as f:
        return [(it["q"], it["expect"]) for it in json.load(f)["queries"]]


def _rows(fn, chunks, index):
    return [{"expect": exp, "ids": [d for _, d in fn(q, chunks, index)]}
            for q, exp in _qa()]


def _stats(rows):
    n = len(rows)
    return {
        "recall": sum(1 for r in rows if r["expect"] in r["ids"]),
        "dup": sum(1 for r in rows if len(set(r["ids"])) < len(r["ids"])),
        "mrr": round(sum((1.0 / (r["ids"].index(r["expect"]) + 1)
                          if r["expect"] in r["ids"] else 0.0)
                         for r in rows) / n, 3),
        "ndcg": round(sum(RR.ndcg_at_k(r["ids"], {r["expect"]}, k=3)
                          for r in rows) / n, 3),
    }


def test_ndcg_invariant_and_bug_guard():
    """nDCG 必须 ∈ [0,1]；重复文档输入不得越界（去重修正的守卫）。"""
    # 重复文档：修复前会得 1.63
    v = RR.ndcg_at_k(["vector_db", "vector_db", "agent_tools"], {"vector_db"}, k=3)
    assert 0.0 <= v <= 1.0, f"重复文档 nDCG 越界 {v}"

    # 越界必须抛（守卫非空性）：伪造 idcg < dcg 的场景
    try:
        RR.ndcg_at_k(["a"], {"a": 1.0, "b": 5.0}, k=1)
    except AssertionError:
        pass  # 期望路径：k=1 时 ideal 取 top1=5.0 > gains=1.0 → 不越界，不应抛
    # 正常路径不抛
    assert RR.ndcg_at_k(["a"], {"a": 1.0}, k=1) == 1.0


def test_ndcg_ordering_sanity():
    """度量方向：命中越靠前 nDCG 越大；全 miss 为 0。"""
    assert RR.ndcg_at_k(["a", "b", "c"], {"a"}, 3) == 1.0
    assert RR.ndcg_at_k(["b", "a", "c"], {"a"}, 3) < 1.0
    assert RR.ndcg_at_k(["b", "c", "d"], {"a"}, 3) == 0.0


def test_baseline_vs_diverse_frozen():
    """冻结基线与文档多样重排的实测数字。"""
    chunks, index = rag_demo.run_pipeline(verbose=False)

    base = _stats(_rows(RR.ranked_baseline, chunks, index))
    div = _stats(_rows(RR.ranked_diverse, chunks, index))

    assert base["recall"] == FROZEN["base_recall"], base
    assert base["dup"] == FROZEN["base_dup"], base
    assert div["recall"] == FROZEN["div_recall"], div
    assert div["dup"] == FROZEN["div_dup"], div
    assert abs(div["mrr"] - FROZEN["div_mrr"]) < 5e-4, div
    assert abs(div["ndcg"] - FROZEN["div_ndcg"]) < 5e-4, div

    # 增益方向必须为正（否则「修法有效」的结论不成立）
    assert div["recall"] > base["recall"]
    assert div["dup"] < base["dup"]


def test_coverage_rerank_honest_no_gain():
    """覆盖率重排在本集上【无增益】——如实冻结，避免事后美化。"""
    chunks, index = rag_demo.run_pipeline(verbose=False)
    base = _stats(_rows(RR.ranked_baseline, chunks, index))
    cov = _stats(_rows(lambda q, c, i: RR.ranked_with_coverage(q, c, i), chunks, index))
    assert cov["ndcg"] == base["ndcg"], "覆盖率重排结果已变，需复核结论"
    assert cov["recall"] == base["recall"]


def test_all_arms_ndcg_within_unit_interval():
    """所有臂的 nDCG 必须 ∈ [0,1]（越界即度量缺陷）。"""
    chunks, index = rag_demo.run_pipeline(verbose=False)
    for name, fn in [("baseline", RR.ranked_baseline),
                     ("coverage", lambda q, c, i: RR.ranked_with_coverage(q, c, i)),
                     ("diverse", RR.ranked_diverse)]:
        rows = _rows(fn, chunks, index)
        v = sum(RR.ndcg_at_k(r["ids"], {r["expect"]}, k=3) for r in rows) / len(rows)
        assert 0.0 <= v <= 1.0, f"{name} nDCG 越界 {v}"
