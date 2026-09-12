#!/usr/bin/env python3
"""Self-RAG 置信门控（B 线）：免检索直答 vs 全检索，量化「检索到底买到什么」。

思想迁移（P2 B-P37 Self-RAG「按需检索」）：检索不是免费的；高置信查询可免检索直答。
本 demo 无生成器，故把「免检索直答」定义为**只用标题索引**作答：不切块、不建 chunk
向量、不做 BM25 融合。可复算数字：

    免检索可答率 = 标题索引 top-1 == 期望文档的题数 / 总题数

在给定题集上校准置信阈值 τ（约束：门控后命中数不降），报告跳过率（省下的检索占比）。

两个题集：
  - golden(5)：冻结基线 EVAL_SET，题目与标题词面高度对齐 → 区分度低
  - W2(32)：bline/w2_golden.json，含 paraphrase/adversarial → 可区分检索增益

上游缺陷（本门先探测后修正）：
  parse_docs 用「首行」当 title，单行 .txt（cooking/travel）退化为 title==body，
  使任何「标题即闭卷」的基线被正文污染（虚高）。本门报告该缺陷并用修正短标题重测。

只读约束：`rag_demo.py` 与 `docs/` 已被 FINAL_ACCEPTANCE 冻结，本门只 import 不改。

运行：python3 tools/confidence_gate.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import rag_demo as R  # noqa: E402

TAU_GRID = [round(0.05 * i, 2) for i in range(0, 21)]  # 0.00 .. 1.00
TITLE_MAX = 24
W2_GOLDEN = os.path.join(ROOT, "bline", "w2_golden.json")


def derive_short_title(doc):
    """修正版短标题：首行按中英冒号/句号/逗号切首段并限长（对单行 txt 亦有效）。"""
    t = doc["title"].strip()
    for sep in ("：", ":", "。", "，", ",", "\n", "；", ";"):
        if sep in t:
            t = t.split(sep)[0]
    return t.strip()[:TITLE_MAX]


def detect_title_defect(docs):
    """返回 title 退化为全文（title==body）的 doc_id 列表。"""
    return [d["doc_id"] for d in docs
            if d["title"].strip() == d["text"].strip()]


def build_title_index(docs, short=True):
    """标题索引：每篇一个「标题字二元组」向量（无正文、无分块）。"""
    titles = [(derive_short_title(d) if short else d["title"]) for d in docs]
    return R.TfidfIndex([R.bigrams(t) for t in titles])


def title_pick(query, docs, tindex):
    """免检索直答：标题索引 top-1。返回 (doc_id | None, 余弦分)。"""
    qvec = tindex._vec(R.bigrams(query))
    if not any(qvec.values()):
        return None, 0.0
    score, i = max(tindex.cosine(qvec))
    if score <= 0:
        return None, 0.0
    return docs[i]["doc_id"], score


def load_w2():
    with open(W2_GOLDEN, encoding="utf-8") as f:
        g = json.load(f)
    return [(it["q"], it["expect"]) for it in g["queries"]]


def calibrate(rows, full_hits):
    """τ 网格扫描：最大化跳过数，约束「门控后命中数 >= 全检索命中数」。"""
    best = None
    for tau in TAU_GRID:
        hits, skipped = 0, 0
        for r in rows:
            if r["score"] >= tau and r["cb"] is not None:
                skipped += 1
                hits += r["cb_hit"]
            else:
                hits += r["full_hit"]
        if hits < full_hits:
            continue
        if best is None or (skipped, tau) > best:
            best = (skipped, tau)
    return best


def run_set(label, qa_pairs, docs, chunks, index, tindex_raw, tindex_fix, show_rows):
    n = len(qa_pairs)
    full_hits = 0
    rows_raw, rows_fix = [], []
    for q, expect in qa_pairs:
        res = R.retrieve(q, chunks, index, top_k=3)
        fh = any(c["doc_id"] == expect for _, c in res)
        full_hits += fh
        for tindex, bucket in ((tindex_raw, rows_raw), (tindex_fix, rows_fix)):
            cb_doc, cb_score = title_pick(q, docs, tindex)
            bucket.append({"q": q, "expect": expect, "cb": cb_doc,
                           "score": cb_score, "cb_hit": cb_doc == expect,
                           "full_hit": fh})

    cb_raw = sum(r["cb_hit"] for r in rows_raw)
    cb_fix = sum(r["cb_hit"] for r in rows_fix)
    cal_raw = calibrate(rows_raw, full_hits)
    cal_fix = calibrate(rows_fix, full_hits)

    print(f"\n{'=' * 62}\n[{label}] n={n}  全检索基线 recall@3 = {full_hits}/{n}"
          f" = {full_hits / n:.3f}")
    print(f"  免检索可答率  未修正title {cb_raw}/{n} = {cb_raw / n:.3f}"
          f"   修正短标题 {cb_fix}/{n} = {cb_fix / n:.3f}")
    if cal_raw:
        print(f"  门控(未修正) τ*={cal_raw[1]:.2f} 跳过 {cal_raw[0]}/{n}"
              f" = {cal_raw[0] / n:.0%}")
    if cal_fix:
        print(f"  门控(修正)   τ*={cal_fix[1]:.2f} 跳过 {cal_fix[0]}/{n}"
              f" = {cal_fix[0] / n:.0%}")

    if show_rows:
        print("  逐题（修正短标题）：")
        for r in rows_fix:
            print(f"    {'HIT ' if r['cb_hit'] else 'miss'}  {r['q'][:22]:24s}"
                  f" 选={str(r['cb']):13s} 期望={r['expect']:13s}"
                  f" 分={r['score']:.3f}")

    return {"n": n, "full_hits": full_hits, "cb_raw": cb_raw, "cb_fix": cb_fix,
            "cal_raw": cal_raw, "cal_fix": cal_fix}


def main():
    docs = R.parse_docs(R.DOCS_DIR)

    defect = detect_title_defect(docs)
    print(f"[缺陷] parse_docs 的 title 退化为全文（title==body）：{defect or '无'}")
    for d in docs:
        st = derive_short_title(d)
        mark = " <-- 退化" if d["doc_id"] in defect else ""
        print(f"   {d['doc_id']:14s} 原title(len={len(d['title']):3d}) "
              f"-> 短标题(len={len(st):2d})「{st}」{mark}")

    tindex_raw = build_title_index(docs, short=False)
    tindex_fix = build_title_index(docs, short=True)
    chunks, index = R.run_pipeline(verbose=False)

    r5 = run_set("golden(5) 冻结 EVAL_SET", list(R.EVAL_SET), docs, chunks, index,
                 tindex_raw, tindex_fix, show_rows=True)
    r32 = run_set("W2(32) 含 paraphrase/adversarial", load_w2(), docs, chunks, index,
                  tindex_raw, tindex_fix, show_rows=False)

    print(f"\n{'=' * 62}\n[结论]")
    print(f"  golden(5)  免检索可答率 {r5['cb_fix']}/{r5['n']}"
          f"  全检索 {r5['full_hits']}/{r5['n']}  → 区分度低（题面贴标题）")
    print(f"  W2(32)     免检索可答率 {r32['cb_fix']}/{r32['n']}"
          f" = {r32['cb_fix'] / r32['n']:.3f}"
          f"  全检索 {r32['full_hits']}/{r32['n']}"
          f" = {r32['full_hits'] / r32['n']:.3f}")
    gain = r32['full_hits'] / r32['n'] - r32['cb_fix'] / r32['n']
    print(f"  → 检索增益（W2 全检索 − 免检索）= {gain:+.3f}"
          f"（{'检索有价值' if gain > 0 else '本集内检索未显增益'}）")

    ok = (r5["full_hits"] == r5["n"]) and r32["cal_fix"] is not None
    print("\nCONFIDENCE GATE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
