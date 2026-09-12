#!/usr/bin/env python3
"""守护门脚本：确保独立门脚本（非 pytest 风格）在测试套件中也被执行。

覆盖：
  - tests/test_metrics_regression.py  → R59 六度量 + 契约 + 上游缺陷 + 非空性
  - tools/confidence_gate.py          → 免检索 vs 全检索 + τ 校准 + title 缺陷

任一脚本 exit != 0 → 本测试 FAIL（fail-closed）。
运行：python3 -m pytest -q tests/
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GATES = [
    ("metrics_regression", ROOT / "tests" / "test_metrics_regression.py"),
    ("confidence_gate", ROOT / "tools" / "confidence_gate.py"),
    ("rerank_ndcg", ROOT / "tools" / "rerank_ndcg.py"),
]


def test_gate_scripts_exit_zero():
    for name, script in GATES:
        assert script.is_file(), f"{name}: 缺脚本 {script}"
        p = subprocess.run([sys.executable, str(script)],
                           capture_output=True, text=True, cwd=str(ROOT))
        assert p.returncode == 0, (
            f"{name} exit={p.returncode}\n--- stdout ---\n{p.stdout[-1500:]}"
            f"\n--- stderr ---\n{p.stderr[-800:]}"
        )
