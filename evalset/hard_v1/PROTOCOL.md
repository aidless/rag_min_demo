# hard_v1 protocol

Frozen before any retrieval score was computed. `SHA256.txt` is the sha256 of the exact bytes of `questions.json`. The runner must refuse to score if that hash does not match, or if any file named in `corpus_sha256` has changed.

This file is the analysis plan. A later commit may add a runner that implements it and the numbers that runner prints. It may not edit `questions.json`, this protocol, or the hash to fit those numbers.

## What this is

Author-written labels over the seven files in `docs/`. Twenty-eight questions: paraphrase 8, exact-entity 8, multi-doc 6, unanswerable 6. They are not user logs, not doubly rated, and not a sample of production traffic. The original five-question quiz in `rag_demo.py` stays a smoke test and is not part of these twenty-eight. It is not used to choose this grid.

No pilot retrieval was run to drop, rewrite, or relabel an item. Before this file was committed, the author read `docs/` and grepped it to check that exact-entity strings sit in the labeled file and that the unanswerable topics are absent. That grep is a label audit, not a score.

## What was already chosen

- Fusion weights 0.6 and 0.4 are the constants already in `rag_demo.py`. They were not fit on these twenty-eight questions.
- Chunk length 220 and overlap 40 are the demo defaults. Lengths 120 and 400 are round neighbors picked in this protocol, not by searching the scores.
- The weight grid is the 0.2 step already used in older notes under `bline/`. It is reported in full. No weight and no chunk length becomes a new default because it looked best here.
- Tuning-on-test: the same frozen file is both the only measurement and the grid that gets looked at. Quoting a champion weight after seeing the table would be a second use of the same items. The win rule below does not permit that.

## Procedures

Overlap is 40. Tokens are `rag_demo.bigrams`. Vectors are `rag_demo.TfidfIndex`. Chunks are `rag_demo.chunk_text`. Before any BM25 call, `rag_demo.chunk_tokens_global` is the token list of the index being queried, because the class reads that global.

**fusion(w)** for w in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}: set `W_COS = w` and `W_BM25 = 1 - w`, call `rag_demo.retrieve(..., top_k=3)`, then restore 0.6 and 0.4. This is cosine top-10 (`CANDIDATES = 10`) followed by weighted normalized fusion. w = 0 reranks that cosine pool by BM25. It is not full-corpus BM25. w = 1 is the cosine order of that pool.

**cosine** (full corpus): `index.cosine`. If the query vector has no positive weight, return nothing. Otherwise sort by score descending, ties by earlier chunk (stable sort). Take three. This is not the same tie-break as fusion: fusion sorts `(score, index)` descending, so equal scores prefer the later chunk. Do not "fix" that after seeing scores.

**bm25** (full corpus): `index.bm25`. If the maximum score is 0, return nothing. Otherwise sort by score descending, ties by earlier chunk. Take three.

## Hit, recall@3, MRR

Answerable items only. Unanswerable items are out of recall and MRR.

- One expected doc: a hit is any returned chunk with that `doc_id`. Rank is the 1-based index of the first such chunk. A miss contributes 0.
- Two expected docs: a hit is every expected id appearing at least once in the returned chunks. Rank is the 1-based index of the shortest prefix that covers them. A miss contributes 0. One doc filling all three slots is a miss. That strict rule is intentional.
- recall@3 = hits / n in the slice. MRR = mean of 1/rank on hits and 0 on misses.

Slices: all answerable items; paraphrase; exact_entity; multi_doc. Order inside a slice is file order.

## Abstain

A query abstains when the procedure returns an empty list. `rag_demo.retrieve` has no score cutoff. It returns nothing only when the query bigram vector has no positive weight. Full-corpus BM25 also returns nothing when every score is 0.

Prediction written before the run: the six unanswerable questions are ordinary words and names, so fusion(0.6) and cosine at chunk 220 are expected to abstain on none of them. Report the count either way. Do not add a threshold after the fact.

Report abstain counts at chunk 220 for fusion(0.6), cosine, and bm25, split by answerable vs unanswerable.

## Bootstrap

B = 2000. Seed = 20261116. Generator = `random.Random`.

Each reported interval uses a fresh `random.Random(20261116)`, so it does not depend on call order. One replicate draws n indices with replacement via `randrange(n)`. For a table cell, recall and MRR share that resample. Each paired difference (fusion minus cosine, fusion minus bm25, bm25 minus cosine; recall and MRR separately) has its own fresh generator and its own resamples.

After sorting replicates ascending:

- `k_lo = (25 * (B - 1)) // 1000`
- `k_hi = (975 * (B - 1) + 999) // 1000`

The interval is `(sorted[k_lo], sorted[k_hi])`.

## Grid

- Chunk lengths 120, 220, 400, overlap 40. Methods: cosine, bm25, fusion(0.6). Overall and per answerable category.
- Weight sweep only at chunk 220: the six fusion weights, overall and per answerable category.
- Primary comparison: chunk 220, all answerable items, cosine vs bm25 vs fusion(0.6).

## Win rule

On that primary slice, fusion(0.6) beats another method on recall only if the recall difference is positive and the paired 95% bootstrap interval lies entirely above 0. The same rule is reported separately for MRR. A higher point estimate whose interval contains 0 is not a win. Beating "the single methods" means beating both cosine and BM25. The demo default stays 220 and 0.6/0.4 whatever the table says.

## Runner discipline

Display rounding is 4 decimal places for rates and intervals, plus the integer hit count. The smoke line (doc count, parsed characters, chunk count, vocab size, smoke recall, smoke MRR) comes from `parse_docs` and `run_pipeline` / `evaluate` on the unchanged demo defaults, after fusion globals are restored to 0.6 and 0.4.

`make check` reruns this protocol and fails if the README number block or the committed results file disagrees, or if the question-file hash or corpus hashes disagree.
