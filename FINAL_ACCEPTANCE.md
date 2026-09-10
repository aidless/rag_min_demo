# FINAL_ACCEPTANCE — B线 rag_min_demo（本地验收，不发布）

- 日期：2026-09-10（复验轮 P2-CLOSE）
- 基线 commit：26f7140d64e45845868f4dc407c5815b0910bb8c（P2-HARDEN）
- 验收结论：全部绿灯，仅本地 commit，未 push，未建远端。

## 1. 基线（冻结，不可改）

- `rag_demo.py` + `docs/` 未动（冻结）。
- 预期冻结数字：recall@3=5/5，MRR=1.000。
- 实测（cwd=rag_min_demo）：
  - `python3 -m pytest -q` → `1 passed in 0.02s`，exit 0。
  - `bash demo.sh` → exit 0，尾部含 `recall@3 = 5/5, MRR = 1.000`（5 题 PASS rank=1）。
  - `python3 rag_demo.py` → exit 0，同上尾部数字。

## 2. W1/W2/W3（cwd=workspace root，前缀 PYTHONDONTWRITEBYTECODE=1）

- `python3 rag_min_demo/bline/w1_harness.py` → exit 0；7 格点全部 recall@3=5/5 MRR=1.000；落盘 bline/w1_results.json。
- `python3 rag_min_demo/bline/w2_harness.py` → exit 0；7 格点全部 recall@3=31/32 MRR=0.969 Wilson95=[0.8426, 0.9945]，W1子集=5/5，分歧 0（p=1.0）；落盘 bline/w2_results.json。
- `python3 rag_min_demo/bline/w3_harness.py` → exit 0；replay 门禁 PASS（7 格点 recall=[31,32] MRR=0.9688）；extended 双口径落盘 bline/w3_p2_check.json。
- 注：三者 stdout 首行均有 `fatal: 不是 git 仓库（或者任何父目录）：.git`——系 harness 在 workspace root 取 git_sha 失败所致，git_sha 记为 unknown，不影响评测数字与 exit 0。

## 3. 迁移边界

- `bline/LICENSE` 首行 `MIT License`（MIT 确认）。
- `git -C rag_min_demo ls-files | grep -E "pycache|venv"` 为空（grep exit 1），无 `__pycache__`/venv 被跟踪（磁盘上有 `rag_min_demo/__pycache__/`，但被 .gitignore 覆盖，未跟踪）。
- `git -C rag_min_demo remote -v` 为空：未建远端。
- 未写核心检索代码，未新建 repo，未碰 A 线（gsm8k-self-evolve / mini-agent-runtime-bench 未动）。

## 4. 本轮改动

- `PHASE_LOG.md`：追加 `## P2-CLOSE`（实际命令输出）。
- `bline/w1_results.json`、`bline/w2_results.json`、`bline/w3_p2_check.json`：harness 重跑仅 `generated_at` 时间戳变化，数字不变，随本轮一并提交（均为已跟踪结果文件，不含新代码）。
- 本文件（FINAL_ACCEPTANCE.md）：新建验收记录。
- Commit（本地，绝不 push）：`P2-CLOSE: baseline + W1-W3 green, no push`。
