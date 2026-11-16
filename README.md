# 字二元组 RAG 教学演示：冒烟跑通，不是检索质量

[![CI](https://github.com/aidless/rag_min_demo/actions/workflows/checks.yml/badge.svg)](https://github.com/aidless/rag_min_demo/actions/workflows/checks.yml)

> **English TL;DR**: This demo chunks a handful of short docs with the standard library and retrieves them by character-bigram cosine and BM25, to show the chain runs offline.
> The original frozen quiz is only a smoke test, not retrieval quality, synonym handling, or adoption.
> The harder questions were written by the author after reading the docs, hashed before scoring; whether fusion beats either single method is only whatever the pre-registered rule prints in the number block.

这条演示用标准库把几篇短文档切块，再用字二元组的余弦和 BM25 做检索，用来证明整条链路能离线跑通。
原来的冻结小测验只说明冒烟是否通过，不能当成检索质量、同义改写能力或别人在用。
更难的一版题是作者读完文档后自己写的，先冻结哈希再计分；融合是否优于单路，只以计分块里的预注册规则为准。

## 复现

```bash
make check
```

`make check` 用 `python3` 离线重算下面计分块的每一个数字。题面哈希或语料哈希变了也会失败。不安装第三方包。题面、协议和哈希在 `evalset/hard_v1/`。失败模式见 `FAILURE_MODES.md`。

<!-- hard-v1-numbers:start -->
hard_v1 questions.json sha256: 4478e43167e17eea623850f15c761e6c1edd4eb0933d1a6c546cb51e28699718
corpus sha256 (frozen in the question file, rechecked):
- agent_tools.md 64c6df24fab00f62497ed9cab4f9763a3e555d895ca7790b69fba01c74074b8d
- chunking.md 084142f2e1ea5c30a44f7b1f3163cc9d029e74103da48cb49f63df2240d6f6b7
- cooking.txt 02b7f91914263beb0807edb8dc49afdaf3fb70b5cc0e92f0e0d957e74d79d7b2
- eval.md 2672866197a423981c8144db47563a368ed3453b9af39f04cee4cb090abaddb5
- rag_overview.md efa1b30c0bb75f94ae4aede33d426daa33c3b5837f5635505f75df5cb7e07d97
- travel.txt b48530cedfd8dcf1afb88734abe31fea689e040fa330bfd401fdd372f9f58803
- vector_db.md 982343faf6331b2bd0ed875e336bf49d5aaeb0a8d27e8a3688c94b29aa57424c
Smoke (original five-question quiz, demo defaults, not part of the 28): docs 7, parsed chars 1411, chunks 10, vocab 1098, recall@3 5/5, MRR 1.0000.
hard_v1 is author-written and small: n=28 (paraphrase 8, exact_entity 8, multi_doc 6, unanswerable 6). Labels were hashed before this scoring run.
Decision (chunk 220, overlap 40, answerable items only):
中文：
- 按预注册规则，fusion(0.6) 的 recall 没有同时胜过 cosine 和 bm25。
- fusion(0.6) 与 cosine 在 recall 上打平（差 0.0000，区间 [0.0000, 0.0000]）。
- fusion(0.6) 与 bm25 在 recall 上打平（差 0.0000，区间 [0.0000, 0.0000]）。
- 按预注册规则，fusion(0.6) 的 MRR 没有同时胜过 cosine 和 bm25。
- fusion(0.6) 在 mrr 上低于 cosine（差 -0.0076，配对区间 [-0.0227, 0.0000]）。
- fusion(0.6) 在 mrr 上的点估计高于 bm25（差 0.0076，配对区间 [0.0000, 0.0227]），但按规则不算赢。
English:
- On recall, fusion(0.6) does not beat both single methods.
- fusion(0.6) ties cosine on recall (diff 0.0000, CI [0.0000, 0.0000]).
- fusion(0.6) ties bm25 on recall (diff 0.0000, CI [0.0000, 0.0000]).
- On MRR, fusion(0.6) does not beat both single methods.
- fusion(0.6) is lower than cosine on mrr (diff -0.0076, paired CI [-0.0227, 0.0000]).
- fusion(0.6) is higher than bm25 on mrr (diff 0.0076, paired CI [0.0000, 0.0227]) but that is not a win.

Primary methods at chunk 220, all answerable items:
| method | hits | n | recall@3 | recall 95% CI | MRR | MRR 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| cosine | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8409 | [0.7500, 0.9318] |
| bm25 | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8258 | [0.7121, 0.9242] |
| fusion_0.6 | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8333 | [0.7273, 0.9318] |

Paired differences at chunk 220, all answerable items (A minus B):
| metric | A | B | diff | paired 95% CI | relation | win |
| --- | --- | --- | --- | --- | --- | --- |
| recall | fusion_0.6 | cosine | 0.0000 | [0.0000, 0.0000] | tie | false |
| mrr | fusion_0.6 | cosine | -0.0076 | [-0.0227, 0.0000] | lower | false |
| recall | fusion_0.6 | bm25 | 0.0000 | [0.0000, 0.0000] | tie | false |
| mrr | fusion_0.6 | bm25 | 0.0076 | [0.0000, 0.0227] | higher | false |
| recall | bm25 | cosine | 0.0000 | [0.0000, 0.0000] | tie | false |
| mrr | bm25 | cosine | -0.0152 | [-0.0379, 0.0000] | lower | false |

Per category at chunk 220:
| category | method | hits | n | recall@3 | recall 95% CI | MRR | MRR 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paraphrase | cosine | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9375 | [0.8125, 1.0000] |
| paraphrase | bm25 | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9167 | [0.7500, 1.0000] |
| paraphrase | fusion_0.6 | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9167 | [0.7500, 1.0000] |
| exact_entity | cosine | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| exact_entity | bm25 | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| exact_entity | fusion_0.6 | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| multi_doc | cosine | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.5000 | [0.5000, 0.5000] |
| multi_doc | bm25 | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.4722 | [0.4167, 0.5000] |
| multi_doc | fusion_0.6 | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.5000 | [0.5000, 0.5000] |

Fusion weight sweep at chunk 220. w_bm25 is IEEE 1-w_cos (1-0.8 = 0.19999999999999996, which is not the literal 0.2). w=0 reranks the cosine top-10; it is not full-corpus BM25.
| w_cos | w_bm25 | category | hits | n | recall@3 | recall 95% CI | MRR | MRR 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 1.0000 | all | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8258 | [0.7121, 0.9242] |
| 0.0 | 1.0000 | paraphrase | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9167 | [0.7500, 1.0000] |
| 0.0 | 1.0000 | exact_entity | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 0.0 | 1.0000 | multi_doc | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.4722 | [0.4167, 0.5000] |
| 0.2 | 0.8000 | all | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8258 | [0.7121, 0.9242] |
| 0.2 | 0.8000 | paraphrase | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9167 | [0.7500, 1.0000] |
| 0.2 | 0.8000 | exact_entity | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 0.2 | 0.8000 | multi_doc | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.4722 | [0.4167, 0.5000] |
| 0.4 | 0.6000 | all | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8333 | [0.7273, 0.9318] |
| 0.4 | 0.6000 | paraphrase | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9167 | [0.7500, 1.0000] |
| 0.4 | 0.6000 | exact_entity | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 0.4 | 0.6000 | multi_doc | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.5000 | [0.5000, 0.5000] |
| 0.6 | 0.4000 | all | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8333 | [0.7273, 0.9318] |
| 0.6 | 0.4000 | paraphrase | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9167 | [0.7500, 1.0000] |
| 0.6 | 0.4000 | exact_entity | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 0.6 | 0.4000 | multi_doc | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.5000 | [0.5000, 0.5000] |
| 0.8 | 0.2000 | all | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8333 | [0.7273, 0.9318] |
| 0.8 | 0.2000 | paraphrase | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9167 | [0.7500, 1.0000] |
| 0.8 | 0.2000 | exact_entity | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 0.8 | 0.2000 | multi_doc | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.5000 | [0.5000, 0.5000] |
| 1.0 | 0.0000 | all | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8409 | [0.7500, 0.9318] |
| 1.0 | 0.0000 | paraphrase | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9375 | [0.8125, 1.0000] |
| 1.0 | 0.0000 | exact_entity | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 1.0 | 0.0000 | multi_doc | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.5000 | [0.5000, 0.5000] |

Chunk length sweep, overlap 40, three methods:
| chunk | category | method | hits | n | recall@3 | recall 95% CI | MRR | MRR 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 120 | all | cosine | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8182 | [0.7045, 0.9242] |
| 120 | all | bm25 | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8182 | [0.7045, 0.9242] |
| 120 | all | fusion_0.6 | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8182 | [0.7045, 0.9242] |
| 120 | paraphrase | cosine | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9375 | [0.8125, 1.0000] |
| 120 | paraphrase | bm25 | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9375 | [0.8125, 1.0000] |
| 120 | paraphrase | fusion_0.6 | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9375 | [0.8125, 1.0000] |
| 120 | exact_entity | cosine | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 120 | exact_entity | bm25 | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 120 | exact_entity | fusion_0.6 | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 120 | multi_doc | cosine | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.4167 | [0.3611, 0.4722] |
| 120 | multi_doc | bm25 | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.4167 | [0.3611, 0.4722] |
| 120 | multi_doc | fusion_0.6 | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.4167 | [0.3611, 0.4722] |
| 220 | all | cosine | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8409 | [0.7500, 0.9318] |
| 220 | all | bm25 | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8258 | [0.7121, 0.9242] |
| 220 | all | fusion_0.6 | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8333 | [0.7273, 0.9318] |
| 220 | paraphrase | cosine | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9375 | [0.8125, 1.0000] |
| 220 | paraphrase | bm25 | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9167 | [0.7500, 1.0000] |
| 220 | paraphrase | fusion_0.6 | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.9167 | [0.7500, 1.0000] |
| 220 | exact_entity | cosine | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 220 | exact_entity | bm25 | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 220 | exact_entity | fusion_0.6 | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 220 | multi_doc | cosine | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.5000 | [0.5000, 0.5000] |
| 220 | multi_doc | bm25 | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.4722 | [0.4167, 0.5000] |
| 220 | multi_doc | fusion_0.6 | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.5000 | [0.5000, 0.5000] |
| 400 | all | cosine | 22 | 22 | 1.0000 | [1.0000, 1.0000] | 0.8106 | [0.7045, 0.9091] |
| 400 | all | bm25 | 21 | 22 | 0.9545 | [0.8636, 1.0000] | 0.8182 | [0.6818, 0.9318] |
| 400 | all | fusion_0.6 | 21 | 22 | 0.9545 | [0.8636, 1.0000] | 0.7955 | [0.6818, 0.9091] |
| 400 | paraphrase | cosine | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 0.8542 | [0.6667, 1.0000] |
| 400 | paraphrase | bm25 | 7 | 8 | 0.8750 | [0.6250, 1.0000] | 0.8750 | [0.6250, 1.0000] |
| 400 | paraphrase | fusion_0.6 | 7 | 8 | 0.8750 | [0.6250, 1.0000] | 0.8125 | [0.5625, 1.0000] |
| 400 | exact_entity | cosine | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 400 | exact_entity | bm25 | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 400 | exact_entity | fusion_0.6 | 8 | 8 | 1.0000 | [1.0000, 1.0000] | 1.0000 | [1.0000, 1.0000] |
| 400 | multi_doc | cosine | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.5000 | [0.5000, 0.5000] |
| 400 | multi_doc | bm25 | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.5000 | [0.5000, 0.5000] |
| 400 | multi_doc | fusion_0.6 | 6 | 6 | 1.0000 | [1.0000, 1.0000] | 0.5000 | [0.5000, 0.5000] |

Abstain at chunk 220. A row abstains only when that procedure returns no chunk. rag_demo.retrieve has no score threshold.
| method | answerable abstain | unanswerable abstain |
| --- | --- | --- |
| cosine | 0/22 | 3/6 |
| bm25 | 0/22 | 3/6 |
| fusion_0.6 | 0/22 | 3/6 |
Pre-registered abstain prediction failed: fusion(0.6) abstained on 3 of 6 unanswerable items. No score threshold was added after the fact.
Empty means the query bigram vector had no positive weight. It is not a score cutoff. Per unanswerable item at chunk 220:
| id | method | abstain | returned docs |
| --- | --- | --- | --- |
| U1 | cosine | false | rag_overview,chunking,agent_tools |
| U1 | bm25 | false | rag_overview,chunking,agent_tools |
| U1 | fusion_0.6 | false | rag_overview,chunking,vector_db |
| U2 | cosine | true | (empty) |
| U2 | bm25 | true | (empty) |
| U2 | fusion_0.6 | true | (empty) |
| U3 | cosine | false | vector_db,agent_tools,chunking |
| U3 | bm25 | false | vector_db,agent_tools,chunking |
| U3 | fusion_0.6 | false | vector_db,vector_db,travel |
| U4 | cosine | false | vector_db,agent_tools,rag_overview |
| U4 | bm25 | false | vector_db,agent_tools,rag_overview |
| U4 | fusion_0.6 | false | vector_db,agent_tools,rag_overview |
| U5 | cosine | true | (empty) |
| U5 | bm25 | true | (empty) |
| U5 | fusion_0.6 | true | (empty) |
| U6 | cosine | true | (empty) |
| U6 | bm25 | true | (empty) |
| U6 | fusion_0.6 | true | (empty) |

Definition, not a fitted result: a two-doc hit's rank is the covering prefix, so that rank is at least 2 and multi_doc MRR is at most 0.5000. 0.5000 on that category means every item completed at rank 2.
Observational only, not the win rule: all-answerable recall is below the slice size at chunk 400 bm25 21/22, chunk 400 fusion_0.6 21/22. The win rule stays chunk 220 and is not moved onto these cells.

Answerable misses at chunk 220 for cosine, bm25, and fusion(0.6): none.

Answerable rows worse than the best rank this metric allows (1 for one doc, 2 for two docs), three methods, all three chunk lengths:
- chunk 120 P2 paraphrase cosine rank 2 (best 1) got eval,chunking,rag_overview expect chunking
- chunk 120 M2 multi_doc cosine rank 3 (best 2) got rag_overview,rag_overview,agent_tools expect rag_overview,agent_tools
- chunk 120 M3 multi_doc cosine rank 3 (best 2) got chunking,chunking,vector_db expect chunking,vector_db
- chunk 120 M5 multi_doc cosine rank 3 (best 2) got vector_db,vector_db,eval expect vector_db,eval
- chunk 120 P2 paraphrase bm25 rank 2 (best 1) got eval,chunking,agent_tools expect chunking
- chunk 120 M2 multi_doc bm25 rank 3 (best 2) got rag_overview,rag_overview,agent_tools expect rag_overview,agent_tools
- chunk 120 M3 multi_doc bm25 rank 3 (best 2) got chunking,chunking,vector_db expect chunking,vector_db
- chunk 120 M5 multi_doc bm25 rank 3 (best 2) got vector_db,vector_db,eval expect vector_db,eval
- chunk 120 P2 paraphrase fusion_0.6 rank 2 (best 1) got eval,chunking,agent_tools expect chunking
- chunk 120 M2 multi_doc fusion_0.6 rank 3 (best 2) got rag_overview,rag_overview,agent_tools expect rag_overview,agent_tools
- chunk 120 M3 multi_doc fusion_0.6 rank 3 (best 2) got chunking,chunking,vector_db expect chunking,vector_db
- chunk 120 M5 multi_doc fusion_0.6 rank 3 (best 2) got vector_db,vector_db,eval expect vector_db,eval
- chunk 220 P2 paraphrase cosine rank 2 (best 1) got eval,chunking,agent_tools expect chunking
- chunk 220 P2 paraphrase bm25 rank 3 (best 1) got eval,agent_tools,chunking expect chunking
- chunk 220 M5 multi_doc bm25 rank 3 (best 2) got vector_db,vector_db,eval expect vector_db,eval
- chunk 220 P2 paraphrase fusion_0.6 rank 3 (best 1) got eval,agent_tools,chunking expect chunking
- chunk 400 P2 paraphrase cosine rank 3 (best 1) got eval,agent_tools,chunking expect chunking
- chunk 400 P5 paraphrase cosine rank 2 (best 1) got chunking,vector_db,agent_tools expect vector_db
- chunk 400 P2 paraphrase bm25 rank 0 (best 1) got eval,agent_tools,rag_overview expect chunking
- chunk 400 P2 paraphrase fusion_0.6 rank 0 (best 1) got eval,agent_tools,rag_overview expect chunking
- chunk 400 P5 paraphrase fusion_0.6 rank 2 (best 1) got chunking,vector_db,agent_tools expect vector_db

Tuning disclosure: 0.6/0.4 and chunk 220 / overlap 40 were already the demo defaults in rag_demo.py before these 28 questions existed. Neighbors 120 and 400 and weights [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] were written into PROTOCOL.md before scoring. No weight or chunk length is promoted to a new default from this grid. Reporting the grid on the same frozen file is a tuning-on-test risk; the win rule is the only comparison that counts.
Bootstrap: B=2000, seed=20261116, fresh random.Random per interval, k_lo=(25*(B-1))//1000=49, k_hi=(975*(B-1)+999)//1000=1950.
<!-- hard-v1-numbers:end -->

## 范围

冒烟测验仍是 `rag_demo.py` 里原来的那一小套题，用来看链路是否跑通，不进入更难题的计分。`bline/` 与 `PHASE_LOG.md` 里的旧笔记不由 `make check` 重算，也不算本页的结论。

## Status

STATUS: WORK-IN-PROGRESS — not A yet

Gap: the live GitHub About text is still the previous one-line description. This pull request cannot change repository settings. It should read: "Minimal zero-dependency character-bigram RAG teaching demo. A smoke pass is not retrieval quality, and the author-written set is not production use or adoption."

缺口：线上 GitHub About 仍是旧的一句话。本 PR 改不到仓库设置。应改为：「最小零依赖字二元组 RAG 教学演示。冒烟跑通不是检索质量，作者自写题集也不等于生产使用或采用。」
