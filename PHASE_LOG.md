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

## P2-CLOSE（B线最终验收：基线 + W1/W2/W3 全绿，仅本地提交，不 push）

- 日期：2026-09-10；基线 commit：26f7140d64e45845868f4dc407c5815b0910bb8c。
- 步骤1 只读检查（cwd=workspace root）：
  - `git -C rag_min_demo status --short` → 空（干净）。
  - `git -C rag_min_demo log --oneline -3` → `26f7140 P2-HARDEN: fix run path + baseline test + demo.sh + CI` / `59e2e7e B-line: rag_min_demo baseline (recall@3=5/5 MRR=1.0, W1-W3 PASS) + MIT LICENSE + gitignore`（仅 2 条）。
- 步骤2 基线（cwd=rag_min_demo）：
  - `python3 -m pytest -q` → `.  [100%] 1 passed in 0.02s`，exit 0（tests/test_baseline.py：hits==5 且 MRR==1.0）。
  - `bash demo.sh` → exit 0；尾部 `=> recall@3 = 5/5, MRR = 1.000`（5 题 PASS rank=1）+ 单查询演示 3 条。
  - `python3 rag_demo.py` → exit 0；同上尾部 `recall@3 = 5/5, MRR = 1.000`。
  - 未动：`rag_demo.py`、`docs/`（冻结）。
- 步骤3 W1/W2/W3（cwd=workspace root，前缀 PYTHONDONTWRITEBYTECODE=1）：
  - `python3 rag_min_demo/bline/w1_harness.py` → exit 0：
    `fatal: 不是 git 仓库（或者任何父目录）：.git`（git_sha 取不到记 unknown，不影响数字）
    `w_cos=0.0/0.2/0.4/0.6/0.8/1.0 + rrf_k=60 全部 recall@3=5/5 MRR=1.000`
    `[W1] 全网格 7 格点已落盘 .../bline/w1_results.json`
  - `python3 rag_min_demo/bline/w2_harness.py` → exit 0：
    `7 格点全部 recall@3=31/32 MRR=0.969 Wilson95=[0.8426, 0.9945] W1子集=5/5`
    `vs ref 全 0 分歧 (b=0,c=0) p=1.0；按文档 ref：rag_overview 5/5, chunking 5/5, vector_db 5/5, eval 4/5, agent_tools 4/4, cooking 4/4, travel 4/4`
    `[W2] 全网格 7 格点已落盘 .../bline/w2_results.json`
  - `python3 rag_min_demo/bline/w3_harness.py` → exit 0：
    `[replay] docs-only冻结管线 × W2-32题：7 格点 recall=[31,32] MRR=0.9688 PASS；门禁 PASS`
    `[extended] docs+corpus扩展管线：doc=30/32 MRR=0.938（w_cos≤0.8）/0.922（w_cos=1.0,rrf），vs ref 全 0 分歧 p=1.0`
    `[W3-P2] 双口径结果已落盘 .../bline/w3_p2_check.json`
  - 冻结基线数字未重写：recall@3=5/5，MRR=1.000 保持原样。
- 步骤4 迁移边界：
  - `bline/LICENSE` 首行 `MIT License`（MIT 确认）。
  - `git -C rag_min_demo ls-files | grep -E "pycache|venv"` → 空（grep exit 1）；磁盘 `rag_min_demo/__pycache__/` 存在但被 .gitignore 覆盖、未跟踪。
  - `git -C rag_min_demo remote -v` → 空（未建远端）。
  - 未新建 repo、未写核心检索代码、未碰 gsm8k-self-evolve / mini-agent-runtime-bench。
- 步骤5 落盘与提交（本地，绝不 push）：
  - 改动：`PHASE_LOG.md` 追加本节；新建 `FINAL_ACCEPTANCE.md`；harness 重跑仅 `generated_at` 时间戳变化的已跟踪结果文件 `bline/w1_results.json`、`bline/w2_results.json`、`bline/w3_p2_check.json`（`git diff` 确认数字无变化）。
  - Commit：`P2-CLOSE: baseline + W1-W3 green, no push`（add 仅上述 5 文件，无新代码文件）。
  - 从未运行 `git push`；无远端可推。
