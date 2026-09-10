# W3-P1 语料快照（SNAPSHOT）

- 日期：2026-09-10；复算：`PYTHONDONTWRITEBYTECODE=1 python3 rag_min_demo/bline/corpus/count_chunks.py`
- 切分规则（与冻结基线同源，脚本直接 import `rag_demo.py` 的 `parse_docs/chunk_text`，
  只读不改）：句子感知（按 。！？ 切）+ 上限 220 字 + 重叠 40 字，超长单句硬切；
  chunk 编号 `doc_id#cN`。
- 总数：**35 chunk**（冻结 `docs/` 10 + 新增 `corpus/` 25），门禁≥30通过，余量+5。

## 来源清单（13个新增文件，全部本仓原创，隨MIT仓，无外部复制）

| 文件 | 主题 | 出处与许可 |
|---|---|---|
| hybrid_retrieval.md | 混合检索 | 本仓原创 |
| rerank.md | 重排 | 本仓原创 |
| embedding_select.md | Embedding选型 | 本仓原创 |
| query_rewrite.md | 查询改写 | 本仓原创 |
| context_manage.md | 上下文管理 | 本仓原创 |
| citation.md | 引用与可追溯 | 本仓原创 |
| judge_eval.md | Judge模型评测 | 本仓原创 |
| index_update.md | 索引更新 | 本仓原创 |
| latency_opt.md | 延迟优化 | 本仓原创 |
| failure_modes.md | RAG失败模式 | 本仓原创 |
| recall_debug.md | 召回排查与监控 | 本仓原创 |
| data_clean.md | 文档清洗 | 本仓原创 |
| fitness.txt | 健身（干扰项，与cooking/travel同类） | 本仓原创 |

## chunk 明细（脚本实测输出）

docs（冻结，10）：agent_tools#c0(193) chunking#c0(213) chunking#c1(81)
cooking#c0(89) eval#c0(165) eval#c1(127) rag_overview#c0(214) travel#c0(83)
vector_db#c0(173) vector_db#c1(193)。

corpus（新增，25）：citation#c0(170) citation#c1(109) context_manage#c0(199)
context_manage#c1(78) data_clean#c0(206) data_clean#c1(74)
embedding_select#c0(217) embedding_select#c1(84) failure_modes#c0(156)
failure_modes#c1(144) fitness#c0(136) hybrid_retrieval#c0(206)
hybrid_retrieval#c1(92) index_update#c0(172) index_update#c1(108)
judge_eval#c0(185) judge_eval#c1(126) latency_opt#c0(213) latency_opt#c1(71)
query_rewrite#c0(219) query_rewrite#c1(73) recall_debug#c0(220)
recall_debug#c1(66) rerank#c0(205) rerank#c1(69)。

`count_chunks.py` 全量表即真相，`total=35`。

## 基线冻结核查

`rag_demo.py` 与 `docs/` 7文件 mtime 仍为 2026-09-09（早于全部新增），
本次只读基线（import 切分函数 + mtime 取证），零修改。
