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

---

## P3-HARDEN-B1：B 线评测门（2026-09-12，本地 commit，绝不 push）

- 前提：`P3-harden-design.md` v0.1 的 B 线 5 配方；用户全权授权；`rag_demo.py` + `docs/` 保持**冻结**未动。
- 新增（全部为**新文件**，不改基线）：
  - `tools/longbench_metrics.py`：LongBench 六度量 stdlib-only 移植（count/retrieval/classification/f1/qa_f1/rouge-l）。
  - `tests/test_metrics_regression.py`：R59 六度量回归门（含契约 + 上游缺陷 + 非空性）。
  - `tools/confidence_gate.py`：Self-RAG 置信门控（免检索直答 vs 全检索 + τ 校准）。
  - `tests/test_confidence_gate.py`：门控数字回归（含污染方向与非空性）。
  - `tools/rerank_ndcg.py`：nDCG@k 度量 + 非 LLM reranker 三臂对照。
  - `tests/test_rerank_ndcg.py`：rerank/nDCG 回归。
  - `tests/test_gates_run.py`：把三个独立门脚本纳入 pytest 守护。
- 实测数字（可复算，cwd=rag_min_demo，`python3`）：
  - `tests/test_metrics_regression.py` → exit 0：count 0.5000 / retrieval 0.5000 / cls 1.0000 / f1 0.8182 / qa_f1 0.8000 / rouge 0.8000（**与 R59 记档六值逐项一致**）。
  - `tools/confidence_gate.py` → exit 0：
    - 缺陷检出：`parse_docs` 的 title 退化为全文（title==body）= `['cooking','travel']`（单行 .txt）。
    - golden(5)：免检索 5/5、全检索 5/5（区分度低）。
    - W2(32)：免检索(未修正) 20/32=0.625 → **免检索(修正) 17/32=0.531**；全检索 31/32=0.969。
    - **检索增益 = +0.438**（修正后）；τ*=0.40 可跳过 15/32（47%）且不丢命中。
  - `tools/rerank_ndcg.py` → exit 0（三臂）：
    - 冻结基线 recall@3=0.969 MRR=0.969 nDCG@3=0.969，**含重复文档名次 18/32**。
    - 覆盖率重排：ΔnDCG +0.0000（**无增益，如实记录**）。
    - 文档多样重排：recall@3 **1.000(+0.031)**、MRR 0.979、nDCG@3 **0.984(+0.0156)**、重复名次 **0/32**。
  - `python3 -m pytest -q tests/` → **12 passed**，exit 0。
  - `python3 rag_demo.py` → `recall@3 = 5/5, MRR = 1.000`（**冻结基线未回归**）。
- 本轮两项**缺陷发现与修正**（非装饰性）：
  1. `classification_score` 边遍历边 `remove` → 跳过待删项；修正为列表推导（反证：上游语义 0.5 vs 正解 1.0）。
  2. nDCG 按 chunk 名次直接计分 → **同一文档多 chunk 被重复计分，nDCG 可 >1（实测 1.63）**；修正为文档级去重 + 越界不变量守卫。
- 未迁移（诚实边界）：
  - LongBench 的 jieba/fuzzywuzzy/rouge 分支（retrieval_zh/rouge_zh/qa_f1_zh/code_sim）——不在 R59 六度量内且违反零依赖约束。
  - RankGPT 的**列式 LLM rerank prompt + LLM 调用**未实现（本仓无 LLM/权重）；仅落地 nDCG 度量与非 LLM 确定性 reranker。
  - StructGPT 表问答分支：B 线**无表格语料**，落地需新增数据（属新增能力而非迁移配方）→ 本轮不做，留待有表格语料时评估。
- 迁移边界复核：未建远端；未 push；未碰 `gsm8k-self-evolve` / `mini-agent-runtime-bench`。


---

## P3-HARDEN-B2：B 线补齐两处「未迁移项」（2026-09-12，本地 commit，绝不 push）

- 动因：P3 首轮把 B 线两处标为「未迁移」——RankGPT 列式 rerank（无 LLM）、
  StructGPT 表问答（无表格语料）。本轮把**可验证的非 LLM 部分**补齐。
- 新增（只加法；`rag_demo.py`/`docs/` 仍冻结）：
  - `tools/rankgpt_rerank.py`：列式排列提示模板 + 排列解析 + 可插拔执行器 + 滑窗策略。
  - `tools/table_qa.py` + `tools/table_fixture.json`：两阶段（schema 选择 → cell 定位）
    表问答 + EM 度量 + 最小确定性 fixture。
  - `tests/test_rankgpt_table.py`：13 项断言（含诚实边界与非空性）。
- 实测（cwd=rag_min_demo，`python3`）：
  - `python3 -m pytest -q tests/` → `25 passed`（原 12 + 新增 13），exit 0。
  - `python3 rag_demo.py` → `recall@3 = 5/5, MRR = 1.000`（冻结基线未回归）。
- 本轮发现并修正的 1 处真缺陷：
  - `table_qa.normalize` 先把 `.` 当标点剔除 → 小数被毁（`"2.0"` 变 `"20"`），
    使 `em("2","2.0")` 误判为不等。修正为**先试数值解析**再退到去标点。
- 新增的守卫（反证性）：
  - 未注入 `llm_fn` 时**显式抛 `NoLLMError`**，拒绝静默退化为原序（防止假结果）。
  - `llm_fn` 输出不可解析为完整排列时**显式抛 ValueError**。
  - **回显提示必须被拒**：提示内联格式示例 `[] > [] > []` 会使数字重复出现，
    解析要求恰为 `1..n` 的完整排列 → 回显必然被拒（fail-closed）。
- 诚实边界（未迁移，保持如实）：
  - **真实 LLM 调用未接入**（本仓无 LLM/权重）：接口就绪，行为已由确定性桩验证；
    这不等于 RankGPT 的 nDCG 增益已被复现。
  - 表问答是**小表确定性管线**（5 行 fixture），不是 LLM 语义解析，不声称泛化性。
