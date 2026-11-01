# 最小 RAG 全链路 demo

一句话：7 个文档 → 解析 → 句子感知分块 → 字二元组 TF-IDF 向量化 →
余弦初排 + BM25 加权融合重排 → golden 集评测。**零依赖**，只有 Python3 标准库。

## 运行（前提：先 `cd` 到本仓根，即 `rag_demo.py` 所在目录）

```bash
python3 rag_demo.py                          # 全链路 + 5 题评测
python3 rag_demo.py --query "分块太大有什么坏处"  # 单次查询演示
```

### 一键复现

```bash
python3 --version   # 需 Python 3.8+（实测 3.12.3）
bash demo.sh        # 等价于 python3 rag_demo.py + 一次单查询演示
```

预期输出（尾部）：`recall@3 = 5/5`，`MRR = 1.000`（5 题全部 PASS 且 rank=1）。

实测：解析 7 文档 1411 字 → 10 chunk → 词表 1098 → 评测 recall@3 = 5/5，MRR = 1.000。

## 链路说明（面试按这个讲）

1. **解析**：读 `.md`/`.txt`，去 markdown 标记，保留 doc_id/标题/原文/字数。
   生产升级：PDF 用 MinerU 解析，HTML 去标签，元数据入库。
2. **分块**：句子感知（按 。！？ 切）+ 上限 220 字 + 重叠 40 字，超长单句硬切。
   面试点：分块太大带噪声浪费窗口，太小切断语义——参数用召回率实测，不拍脑袋。
3. **向量化**：字二元组 TF-IDF（`1+log(tf)`，平滑 idf），L2 归一化。
   面试点：归一化后余弦才可比；中文不用分词器也能 work，但生产用 embedding
   （BGE/M3E）+ FAISS 才能处理同义改写，这是本 demo 明确不装的部分。
4. **检索排序**：余弦取 top-10 初排 → `0.6*余弦 + 0.4*BM25` 融合重排取 3。
   面试点：纯向量对精确实体翻车，线上都是混合检索；权重本身也是可调参。
5. **评测**：5 条问题各有期望文档（golden 集，冻结），报 recall@3 与 MRR。

## 证明什么 / 不证明什么（面试官追问时主动说）

**证明什么（本仓内可复现）：**

1. **全链路跑通**：7 文档 1411 字 → 10 chunk → 字二元组词表 1098 →
   `recall@3 = 5/5`、`MRR = 1.000`（5 题全部 PASS 且 rank=1）。这是**冻结基线**
   （`tests/test_baseline.py` 断言 `hits==5` 且 `mrr≈1.0`，任一漂移即 FAIL）。
2. **检索有区分度**：语料含 2 个干扰项（`docs/cooking.txt`、`docs/travel.txt`），
   5 题各自命中期望文档而非干扰项——证明检索不是把所有问题都映射到同一文档。
3. **确定性**：`python3 rag_demo.py` 每次输出相同的 5/5 与 1.000（无 LLM、无随机）。
4. **工具与测试可运行**：`python3 -m pytest -q tests/` → **25 passed**（含
   `test_baseline`、`test_confidence_gate`、`test_rerank_ndcg`、`test_rankgpt_table`、
   `test_gates_run`）。`tools/rerank_ndcg.py`、`tools/confidence_gate.py`、
   `tools/rankgpt_rerank.py`（库）+ `tools/table_qa.py` 均可独立运行，exit 0。

**不证明什么（诚实边界）：**

- **不证明检索质量**：语料只有 7 文档 10 chunk，5/5 只说明链路跑通，不说明检索
  泛化质量——规模证据在 `paper_agent` 的 54 万篇 FAISS 侧。
- **字二元组处理不了同义改写**（如问"切分"搜"分块"会弱），生产切 embedding。
- **单机暴力检索**，10 chunk 谈不上 ANN；索引选型结论见 `docs/vector_db.md`。
- **RankGPT 的 nDCG 增益未复现**：`tools/rankgpt_rerank.py` 只落地提示模板 +
  排列解析 + 确定性桩，**未接真实 LLM**（本仓无 LLM/权重）。未注入 `llm_fn` 时
  显式抛 `NoLLMError`，绝不静默退化为原序（fail-closed，已被测试断言）。
- **表问答是 5 行 fixture 的确定性管线**，不是 LLM 语义解析，不声称泛化性。
- **`tools/rerank_ndcg.py` 的文档多样重排增益（recall 31/32→32/32，MRR 0.969→0.979，
  nDCG@3 0.969→0.984）** 只在 W2(32) 这一冻结子集上实测，不外推到其他语料。

## 复算（所有数字从本仓重算）

```bash
# 全链路 + 冻结基线（recall@3=5/5, MRR=1.000）
python3 rag_demo.py

# 测试套件（25 passed）
python3 -m pytest -q tests/

# rerank/nDCG 三臂对照（文档多样重排 Δrecall +0.0312）
python3 tools/rerank_ndcg.py

# 检索增益对照（W2 全检索 − 免检索 = +0.438）
python3 tools/confidence_gate.py
```

## 文件

- `rag_demo.py`：单文件全链路（约 200 行，可现场逐段讲，只读冻结，不改）
- `docs/`：5 个主题文档 + 2 个干扰项（做饭/旅行，证明检索有区分度，只读冻结）
- `bline/`：B 线迁移准备（MIT LICENSE + BLINE_REPORT 设计报告，只新增不改基线）

## B 线（enterprise-rag）迁移准备（bounded，只新增 + 文档）

- 基线冻结：recall@3 = 5/5，MRR = 1.000（本节不改该数字）。
- 脱敏：`rag_min_demo/` 内无真实密钥、无 `/data1` 与 Administrator 绝对路径残留、
  无 TOKEN/API_KEY 变量名、无 symlink、无 venv（2026-09-10 grep + find 检查通过）。
- 许可：`bline/LICENSE` 为 MIT（新仓根 LICENSE 由此复制）。
- 新仓纳入（计划）：config / hybrid / reranker / harness / latency / golden /
  results / demo.sh / BLINE_REPORT；排除：venv、自指 symlink、云凭证、向量库服务。
- 状态：**B 线仓 `../enterprise-rag` 已于 2026-10-11 落地**。大库口径实测
  （11994 chunk）recall@3 = 8/10、MRR = 0.55、p50 = 3.25ms，与本仓的 5/5
  **不可比**：库大小差约 1200 倍，且本仓语料只有 `.md`，工程事实
  （`.py`/`.json`）系统性缺席。迁移中的三条反直觉结论（真语义向量在同一
  语料上没赢稀疏融合、bigram 重排零贡献、照搬 w_cos = 0.6 反而更差）见
  `../enterprise-rag/hybrid/RESULTS_M1.md`；`enterprise-rag/baseline/` 是本仓
  的冻结复制，逻辑与语料未改。
- 许可：MIT（见 `LICENSE`，与 `bline/LICENSE` 一致）。
