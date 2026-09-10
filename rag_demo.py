#!/usr/bin/env python3
"""最小 RAG 全链路 demo：解析 -> 分块 -> 向量化 -> 检索排序（含评测）。

零依赖（只用标准库），中文用字二元组做向量化。
  全链路 + 评测：  python3 rag_demo.py
  单次查询演示：  python3 rag_demo.py --query "分块太大有什么坏处"
"""
import argparse
import math
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(HERE, "docs")

# 参数：分块上限 / 重叠 / 初排取数 / 融合权重 / 最终取数
MAX_CHARS = 220
OVERLAP = 40
CANDIDATES = 10
W_COS, W_BM25 = 0.6, 0.4
TOP_K = 3

# 5 条评测问题（golden 集：期望命中的文档）
EVAL_SET = [
    ("RAG 由哪几部分组成", "rag_overview"),
    ("分块太大或太小各有什么影响", "chunking"),
    ("向量数据库怎么选型和调优", "vector_db"),
    ("如何评估问答系统的效果", "eval"),
    ("番茄炒蛋怎么做", "cooking"),
]


# ---------- Stage 1：解析 ----------
def parse_docs(docs_dir):
    docs = []
    for fname in sorted(os.listdir(docs_dir)):
        if not (fname.endswith(".md") or fname.endswith(".txt")):
            continue
        with open(os.path.join(docs_dir, fname), encoding="utf-8") as f:
            raw = f.read()
        text = re.sub(r"^#+\s*", "", raw, flags=re.M)
        text = re.sub(r"[*_`>#\-+]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        first = raw.strip().splitlines()[0].lstrip("# ").strip() if raw.strip() else fname
        docs.append({"doc_id": os.path.splitext(fname)[0], "title": first,
                     "text": text, "chars": len(text)})
    return docs


# ---------- Stage 2：分块（句子感知 + 长度上限 + 重叠） ----------
_SENT = re.compile(r"(?<=[。！？!?\n])")


def chunk_text(text, max_chars=MAX_CHARS, overlap=OVERLAP):
    sents = [s.strip() for s in _SENT.split(text) if s.strip()]
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) <= max_chars:
            cur += s
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if overlap and cur else ""
            cur = tail + s
            while len(cur) > max_chars:          # 单句超长则硬切
                chunks.append(cur[:max_chars])
                cur = cur[max_chars - overlap:]
    if cur:
        chunks.append(cur)
    return chunks


# ---------- Stage 3：向量化（字二元组 TF-IDF，L2 归一化） ----------
def bigrams(text):
    t = re.sub(r"\s+", "", text)
    return [t[i:i + 2] for i in range(len(t) - 1)]


class TfidfIndex:
    def __init__(self, chunk_tokens):
        n = len(chunk_tokens)
        df = Counter()
        for toks in chunk_tokens:
            for t in set(toks):
                df[t] += 1
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.vocab_size = len(df)
        self.vecs = [self._vec(toks) for toks in chunk_tokens]
        self.lens = [len(t) for t in chunk_tokens]
        self.avg_len = sum(self.lens) / n

    def _vec(self, toks):
        v, norm = {}, 0.0
        tf = Counter(toks)
        for t, c in tf.items():
            w = (1.0 + math.log(c)) * self.idf.get(t, 0.0)
            v[t] = w
            norm += w * w
        norm = math.sqrt(norm) or 1.0
        return {t: w / norm for t, w in v.items()}

    def cosine(self, qvec):
        out = []
        for i, v in enumerate(self.vecs):
            s = sum(qvec[t] * v.get(t, 0.0) for t in qvec)
            out.append((s, i))
        return out

    def bm25(self, qtoks, k1=1.5, b=0.75):
        qtf = Counter(qtoks)
        out = []
        for i, toks in enumerate(chunk_tokens_global):
            tf = Counter(toks)
            dl = self.lens[i]
            s = 0.0
            for t, qc in qtf.items():
                c = tf.get(t, 0)
                if not c:
                    continue
                idf = self.idf.get(t, 0.0)
                denom = c + k1 * (1 - b + b * dl / self.avg_len)
                s += idf * (c * (k1 + 1) / denom)
            out.append((s, i))
        return out


chunk_tokens_global = []   # 供 BM25 复用（保持单文件可读性）


# ---------- Stage 4：检索排序（余弦初排 -> 与 BM25 加权融合重排） ----------
def retrieve(query, chunks, index, top_k=TOP_K):
    qtoks = bigrams(query)
    qvec = index._vec(qtoks)
    if not any(qvec.values()):
        return []
    cos = {i: s for s, i in index.cosine(qvec)}
    bm = {i: s for s, i in index.bm25(qtoks)}
    cand = sorted(cos, key=cos.get, reverse=True)[:CANDIDATES]
    cmax = max(cos[i] for i in cand) or 1.0
    bmax = max(bm[i] for i in cand) or 1.0
    fused = [(W_COS * cos[i] / cmax + W_BM25 * bm[i] / bmax, i) for i in cand]
    fused.sort(reverse=True)
    return [(s, chunks[i]) for s, i in fused[:top_k]]


def run_pipeline(verbose=True):
    docs = parse_docs(DOCS_DIR)
    chunks, chunk_doc = [], []
    for d in docs:
        for j, c in enumerate(chunk_text(d["text"])):
            chunks.append({"chunk_id": f"{d['doc_id']}#c{j}", "doc_id": d["doc_id"],
                           "title": d["title"], "text": c, "chars": len(c)})
    toks = [bigrams(c["text"]) for c in chunks]
    global chunk_tokens_global
    chunk_tokens_global = toks
    index = TfidfIndex(toks)
    if verbose:
        print(f"[解析] {len(docs)} 个文档，共 {sum(d['chars'] for d in docs)} 字")
        print(f"[分块] {len(chunks)} 个 chunk（上限{MAX_CHARS}字/重叠{OVERLAP}字）")
        for c in chunks:
            print(f"   {c['chunk_id']}  {c['chars']}字")
        print(f"[向量化] 字二元组词表 {index.vocab_size}，向量已 L2 归一化")
        print(f"[检索] 余弦初排 top-{CANDIDATES} -> 0.6*余弦+0.4*BM25 融合重排取 {TOP_K}")
    return chunks, index


def evaluate(chunks, index):
    print("\n[评测] golden 集 recall@3 / MRR：")
    hits, rr = 0, 0.0
    for q, expect in EVAL_SET:
        res = retrieve(q, chunks, index, top_k=3)
        rank = next((i + 1 for i, (_, c) in enumerate(res) if c["doc_id"] == expect), None)
        ok = rank is not None
        hits += ok
        rr += 1.0 / rank if rank else 0.0
        got = res[0][1]["doc_id"] if res else "-"
        print(f"   {'PASS' if ok else 'FAIL'}  Q={q}  期望={expect}  首命中={got}"
              + (f"  rank={rank}" if rank else ""))
    print(f"   => recall@3 = {hits}/{len(EVAL_SET)}, MRR = {rr / len(EVAL_SET):.3f}")
    return hits, rr / len(EVAL_SET)


def main():
    ap = argparse.ArgumentParser(description="最小 RAG 全链路 demo")
    ap.add_argument("--query", help="单次查询演示")
    args = ap.parse_args()
    chunks, index = run_pipeline(verbose=not args.query)
    if args.query:
        res = retrieve(args.query, chunks, index)
        if not res:
            print("查询词全是生僻二元组，无向量表示，返回空。")
            return
        print(f"Q: {args.query}\n")
        for s, c in res:
            print(f"[score={s:.3f}] {c['chunk_id']}《{c['title']}》")
            print(f"  {c['text'][:150]}...\n")
    else:
        evaluate(chunks, index)


if __name__ == "__main__":
    sys.exit(main())
