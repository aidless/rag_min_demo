# Failure modes

Measured figures live only in the README number block. `make check` recomputes that block. This file states the limits and does not copy those figures, so it cannot drift from them.

## 小语料

Seven short teaching files. A perfect smoke quiz, or a perfect recall on the author-written file, does not measure retrieval quality, production behavior, or adoption. There are no users in this repo.

## 作者自写的题

One person read `docs/` and wrote both the questions and the labels, then hashed the file before scoring. There is no user log and no second rater. The set is small. Paraphrases are whatever that author thought a paraphrase was; shared character pairs can remain, so a hit does not mean character bigrams understand synonyms.

## 字二元组

Cosine and BM25 tokenize the same character pairs. They often agree. A rare Latin or Chinese string is easy for both. A wording that shares no in-vocabulary pair is invisible to both, which is not the same thing as a semantic miss.

## 没有拒答阈值

`rag_demo.retrieve` has no score cutoff. It returns nothing only when the query bigram vector has no positive weight. Full-corpus BM25, as defined in `evalset/hard_v1/PROTOCOL.md`, also returns nothing when every score is zero. An empty list is an out-of-vocabulary vector, not a calibrated "I don't know." The protocol predicted the unanswerable items would not come back empty, because they are ordinary words. The number block records whether that prediction held. No threshold was added afterwards.

Unanswerable items that still share a character pair are returned as ordinary top hits. That is a false answer, not abstention.

## 融合不是全库混合检索

Fusion reranks the cosine candidate pool inside `rag_demo.retrieve`. A fusion weight of zero is BM25 inside that pool, not BM25 over every chunk. The full-corpus BM25 column is a different procedure. Do not describe the demo as a complete hybrid search stack.

## 多文档的 MRR 上限

For a two-document label, rank is the length of the shortest prefix that covers both documents. That rank cannot be 1, so the category MRR cannot exceed one half. A value at that cap means every such item completed at rank 2. It is not a mediocre single-document score.

## 召回打满

If every item in a slice hits, the recall interval collapses to a point. That does not estimate any larger population. The pre-registered win rule looks only at the demo chunk length on all answerable items. Other cells in the grid, including places where recall is not full, are observational. They are not a reason to change the default weight or chunk length after seeing the table.

## 在同一套题上网格

The fusion weights and the demo chunk length were already constants in `rag_demo.py` before these questions existed. The neighboring chunk lengths and the weight grid were written into `PROTOCOL.md` before scoring. The grid is still reported on the same frozen file. Choosing a new default from that table would be tuning on the test. This repo does not do that. The only "beats" claim is the one the pre-registered rule prints in the number block.

## 旧笔记

`bline/` and `PHASE_LOG.md` are earlier lab notes. `make check` does not recompute them. They are not conclusions of the README. Figures in those notes that point at another repository are not evidence this clone can rerun.
