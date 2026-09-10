# 最小 RAG 全链路 demo

一句话：7 个文档 → 解析 → 句子感知分块 → 字二元组 TF-IDF 向量化 →
余弦初排 + BM25 加权融合重排 → golden 集评测。**零依赖**，只有 Python3 标准库。

## 运行

```bash
python3 rag_min_demo/rag_demo.py                          # 全链路 + 5 题评测
python3 rag_min_demo/rag_demo.py --query "分块太大有什么坏处"  # 单次查询演示
```

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

## 诚实边界（面试官追问时主动说）

- 语料只有 7 文档 10 chunk，5/5 说明链路跑通，不说明检索质量——规模证据
  在 paper_agent 的 54 万篇 FAISS 侧。
- 字二元组处理不了同义改写（如问"切分"搜"分块"会弱），生产切 embedding。
- 单机暴力检索，10 chunk 谈不上 ANN；索引选型结论见 `docs/vector_db.md`。

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
- 状态：本地准备就绪，可建仓迁移；本次不建远端仓、不写核心检索代码。
  详情见 `bline/BLINE_REPORT.md`。
