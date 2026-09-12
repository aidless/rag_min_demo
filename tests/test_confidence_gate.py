#!/usr/bin/env python3
"""B 线置信门控回归测试（stdlib-only）。

冻结 tools/confidence_gate.py 的关键数字，任何漂移 → FAIL。

本轮固化值（2026-09-10 P3-HARDEN B1）：
  缺陷：parse_docs 的 title 退化为全文的文档 = ['cooking', 'travel']
  golden(5)  免检索 5/5，全检索 5/5（区分度低）
  W2(32)     免检索(未修正) 20/32=0.625  →  免检索(修正) 17/32=0.531
             全检索 31/32=0.969；检索增益 +0.438
  修正后 τ*=0.40 跳过 15/32（47%）

只读：不修改 rag_demo.py / docs/（FINAL_ACCEPTANCE 冻结）。
运行：python3 -m pytest -q tests/test_confidence_gate.py
"""

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 以文件路径加载 tools/confidence_gate.py（tools 非包）
_spec = importlib.util.spec_from_file_location(
    "confidence_gate", str(ROOT / "tools" / "confidence_gate.py")
)
CG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CG)

import rag_demo  # noqa: E402

FROZEN_DEFECT = ["cooking", "travel"]
FROZEN = {
    "golden_n": 5, "golden_cb": 5, "golden_full": 5,
    "w2_n": 32, "w2_cb_fixed": 17, "w2_cb_raw": 20, "w2_full": 31,
}


def _setup():
    docs = rag_demo.parse_docs(rag_demo.DOCS_DIR)
    idx_raw = CG.build_title_index(docs, short=False)
    idx_fix = CG.build_title_index(docs, short=True)
    chunks, index = rag_demo.run_pipeline(verbose=False)
    return docs, idx_raw, idx_fix, chunks, index


def _measure(qa, docs, chunks, index, tindex):
    for q, expect in qa:
        res = rag_demo.retrieve(q, chunks, index, top_k=3)
        full = any(c["doc_id"] == expect for _, c in res)
        cb, score = CG.title_pick(q, docs, tindex)
        yield {"cb_hit": cb == expect, "full_hit": full, "score": score, "cb": cb}


def test_title_defect_detected():
    """上游 title 派生缺陷必须被检出（单行 .txt 的 title==body）。"""
    docs = rag_demo.parse_docs(rag_demo.DOCS_DIR)
    assert CG.detect_title_defect(docs) == FROZEN_DEFECT


def test_short_title_fix():
    """修正短标题：退化文档被截短，正常文档不变形。"""
    docs = {d["doc_id"]: d for d in rag_demo.parse_docs(rag_demo.DOCS_DIR)}
    assert CG.derive_short_title(docs["cooking"]) == "番茄炒蛋的做法"
    assert CG.derive_short_title(docs["travel"]) == "三天两晚的城市旅行建议"
    assert CG.derive_short_title(docs["vector_db"]) == "向量数据库选型与调优"


def test_golden_and_w2_frozen_numbers():
    """冻结两题集的可复算数字。"""
    docs, idx_raw, idx_fix, chunks, index = _setup()

    r5_fix = list(_measure(list(rag_demo.EVAL_SET), docs, chunks, index, idx_fix))
    assert len(r5_fix) == FROZEN["golden_n"]
    assert sum(r["cb_hit"] for r in r5_fix) == FROZEN["golden_cb"]
    assert sum(r["full_hit"] for r in r5_fix) == FROZEN["golden_full"]

    w2 = CG.load_w2()
    assert len(w2) == FROZEN["w2_n"]
    r32_fix = list(_measure(w2, docs, chunks, index, idx_fix))
    r32_raw = list(_measure(w2, docs, chunks, index, idx_raw))
    assert sum(r["cb_hit"] for r in r32_fix) == FROZEN["w2_cb_fixed"]
    assert sum(r["cb_hit"] for r in r32_raw) == FROZEN["w2_cb_raw"]
    assert sum(r["full_hit"] for r in r32_fix) == FROZEN["w2_full"]

    # 污染方向：未修正(含正文泄漏) 应 >= 修正后，否则修正逻辑可疑
    assert FROZEN["w2_cb_raw"] >= FROZEN["w2_cb_fixed"], "title=body 污染方向异常"

    # 检索增益为正：检索确有价值
    assert FROZEN["w2_full"] > FROZEN["w2_cb_fixed"]


def test_calibration_direction():
    """τ 校准：修正后应可行，且跳过数 > 0；且约束下不丢命中。"""
    docs, idx_raw, idx_fix, chunks, index = _setup()
    w2 = CG.load_w2()
    r32 = list(_measure(w2, docs, chunks, index, idx_fix))
    full_hits = sum(r["full_hit"] for r in r32)
    cal = CG.calibrate(r32, full_hits)
    assert cal is not None, "修正后校准不可行"
    skip, tau = cal
    assert skip > 0
    assert 0.0 <= tau <= 1.0


def test_gate_nonvacuity():
    """门自身非空：伪造全 miss 的行，校准必须给出 skip=0（不得虚假跳过）。"""
    rows = [{"q": f"q{i}", "expect": "d", "cb": "other", "score": 0.9,
             "cb_hit": False, "full_hit": True} for i in range(5)]
    cal = CG.calibrate(rows, full_hits=5)
    assert cal is not None
    # score=0.9 高，但 cb_hit 全错 → 若按 score 跳过会丢命中，故只能 skip=0
    assert cal[0] == 0, f"非空性失守：skip={cal[0]}"
