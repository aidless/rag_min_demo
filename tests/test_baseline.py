"""基线守护测试：跑 rag_demo 全链路，断言 recall@3==5/5 且 MRR==1.0。

只读：仅 import rag_demo 并调用其公开函数（run_pipeline / evaluate），
不修改 rag_demo.py / docs/。标准库零依赖（不 import bline 下任何模块）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag_demo


def test_baseline_recall_and_mrr():
    chunks, index = rag_demo.run_pipeline(verbose=False)
    assert len(chunks) > 0
    assert len(rag_demo.EVAL_SET) == 5
    hits, mrr = rag_demo.evaluate(chunks, index)
    assert hits == 5
    assert abs(mrr - 1.0) < 1e-9
