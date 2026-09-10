# P2 PHASE_LOG — B线 rag_min_demo（世界先进封顶）

- 结果 vs 验收 G2：基线 `rag_demo.py` recall@3=5/5 MRR=1.000（重跑一致）；W1 全网格 7 格点 PASS；W2 分歧 0 PASS；W3 longq/p3b（320行零分歧）/p3c（11行零分歧）/termsat 全 PASS。根 `LICENSE`（与 bline 逐字节一致）+ `.gitignore` 已补。G2 通过。
- 产出物：`rag_min_demo/LICENSE`、`rag_min_demo/.gitignore`。
- 失败+纠正：无（一次通过）。B线绝对路径 grep 命中仅为报告文档自述检查方法的字样，非泄露。
- 未动：`rag_demo.py`、`docs/`（冻结）；未建远端仓（需另行授权）。

## P2-HARDEN（质量加固：run path 修复 + 基线测试 + demo.sh + CI）
- Bug 修复：根 README 运行命令 `python3 rag_min_demo/rag_demo.py` 在仓根执行时报
  `No such file or directory`（本仓根即 rag_min_demo/，无嵌套子目录），已改为仓根相对
  路径 `python3 rag_demo.py`（加 `cd 仓根` 前提）；新增"一键复现"小节（含
  `python3 --version` 前提 Python 3.8+、预期输出 recall@3=5/5 MRR=1.000）。
- 新增 `tests/test_baseline.py`：import rag_demo 跑全链路（run_pipeline +
  evaluate），断言 hits==5 且 MRR==1.0；只读，不改 rag_demo.py/docs/；标准库零依赖。
- 新增 `demo.sh`（chmod +x）：`python3 rag_demo.py` + 单查询演示，实测 exit=0。
- 新增 `.github/workflows/verify.yml`：CI 跑 `pip install pytest` → `pytest -q` →
  `python rag_demo.py`（测试自带 sys.path，不依赖 bline import 路径）。
- bline/__pycache__ 检查：`git ls-files | grep pycache` 无命中，未被跟踪，无需
  git rm --cached（.gitignore 已覆盖 __pycache__/ + *.pyc）。
- 全量验证（2026-09-10，Python 3.12.3）：pytest `1 passed`；`bash demo.sh` exit=0，
  尾部 `recall@3 = 5/5, MRR = 1.000`（5 题 PASS rank=1）；`python3 rag_demo.py` 同基线；
  旧错误路径 `python3 rag_min_demo/rag_demo.py` 确认报 No such file（bug 证实已修）。
- 未动：`rag_demo.py`、`docs/`、`bline/` 下既有文件（冻结）；绝不 push（仅本地 commit）。
