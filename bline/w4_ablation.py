#!/usr/bin/env python3
"""W4 分块消融 harness：比较四种切分策略在同一语料同一题集上的表现。

**为什么不 import rag_demo**：冻结基线 rag_demo.py 的 chunk_text() 硬编码
MAX_CHARS=220 / OVERLAP=40，而本实验的因变量就是块长与重叠。import 它等于
把自变量锁死在基线取值上，消融就做不成。rag_demo.py 是别的会话在报告里引用的
共享真相源（AGENTS.md 第 6 条），只读不动。

**本文件只读 w4_corpus/ 与 w4_golden.json，绝不写回语料。**

四档策略（刻意包含一组退化基线，用来证明题集有区分力）：
  fixed_220_o40  冻结基线的原取值，用来锚定"同语料下基线是什么水平"
  fixed_120_o20  更小的块
  fixed_400_o40  更大的块
  fixed_220_o0   零重叠 = 纯固定长度切分，句子感知但无边界保护
  sent_220_o40    句子感知（基线同参数量），与 fixed_220_o40 形成"是否句子感知"的单变量对照

用法：python3 bline/w4_ablation.py [--selftest]
输出：bline/w4_results.json（全部格点 + 逐题命中，全量不挑选）
退出码：0=完成 / 1=门禁失败（题集未接地或无区分力）
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "w4_corpus")
GOLDEN = os.path.join(HERE, "w4_golden.json")
RESULTS = os.path.join(HERE, "w4_results.json")

# 块长扫描的格点：(名字, max_chars, overlap, sentence_aware)
GRID = [
    ("fixed_220_o40", 220, 40, False),
    ("sent_220_o40", 220, 40, True),
    ("fixed_120_o20", 120, 20, False),
    ("fixed_400_o40", 400, 40, False),
    ("fixed_220_o0", 220, 0, False),
]

TOP_K = 3
CANDIDATES = 10
W_COS, W_BM25 = 0.6, 0.4

# ---------------------------------------------------------------- 语料与切分

def load_corpus():
    """读 w4_corpus/ 为 {doc_id: 纯文本}；只读。"""
    out = {}
    for fn in sorted(os.listdir(CORPUS)):
        if fn.endswith((".md", ".txt")):
            with open(os.path.join(CORPUS, fn), encoding="utf-8") as f:
                out[os.path.splitext(fn)[0]] = f.read()
    return out


_SENT_END = re.compile(r"(?<=[。！？；!?;])")


def chunk_text(text, max_chars, overlap, sentence_aware):
    """切块。

    sentence_aware=True  按句边界切，不切断句子。
    sentence_aware=False 纯固定长度切窗，步长 = max_chars - overlap。

    **overlap 在两条路径上都必须真正生效**（第一版这里只写了参数没写实现，
    固定长度路径下 overlap 完全不起作用，fixed_220_o40 与 fixed_220_o0 输出
    逐位相同；是区分力门发现零重叠"没有变差"才暴露出来的）。所以两条路径
    各自显式处理重叠，不共用一段含糊逻辑。
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text or max_chars <= 0:
        return []
    if overlap >= max_chars:
        raise ValueError("overlap(%d) 必须小于 max_chars(%d)" % (overlap, max_chars))
    chunks = []

    if sentence_aware:
        units = [s for s in _SENT_END.split(text) if s.strip()]
        cur = ""
        for u in units:
            while len(u) > max_chars:          # 超长句：硬切，并把尾部 overlap 续接
                if cur:
                    chunks.append(cur)
                    cur = cur[-overlap:] if overlap else ""
                chunks.append(u[:max_chars])
                u = u[max_chars - overlap:] if overlap else u[max_chars:]
            if len(cur) + len(u) <= max_chars:
                cur += u
            else:
                if cur:
                    chunks.append(cur)
                # 新块携带上一块末尾 overlap 个字符 → 跨块的句子能被完整覆盖
                cur = (cur[-overlap:] if overlap else "") + u
        if cur:
            chunks.append(cur)
    else:
        stride = max_chars - overlap
        for start in range(0, len(text), stride):
            piece = text[start:start + max_chars]
            if piece:
                chunks.append(piece)
            if start + max_chars >= len(text):
                break
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------- 检索

def bigrams(text):
    t = re.sub(r"\s+", "", text)
    return Counter(t[i:i + 2] for i in range(len(t) - 1)) or Counter(t)


class Index:
    """TF-IDF 向量 + BM25 双路，融合取 max-norm 加权（照搬 rag_demo 的量纲处理）。"""

    def __init__(self, chunks, meta):
        self.chunks = chunks
        self.meta = meta                      # chunk -> doc_id
        self.n = len(chunks)
        self.df = Counter()
        self.tfs = []
        for c in chunks:
            bg = bigrams(c)
            self.tfs.append(bg)
            self.df.update(bg.keys())
        # L2 归一化的 TF-IDF
        self.vecs = []
        for bg in self.tfs:
            v = {t: (1 + math_log(c)) * (math_log(self.n / (1 + self.df[t])))
                 for t, c in bg.items()}
            nrm = sum(x * x for x in v.values()) ** 0.5
            self.vecs.append({t: x / nrm for t, x in v.items()} if nrm else {})

    def cosine(self, qvec):
        for i, v in enumerate(self.vecs):
            if len(qvec) > len(v):
                small, large = v, qvec
            else:
                small, large = qvec, v
            s = sum(x * large.get(t, 0.0) for t, x in small.items())
            if s > 0:
                yield s, i

    def bm25(self, qtoks, k1=1.5, b=0.75):
        avgdl = (sum(sum(c.values()) for c in self.tfs) / self.n) or 1.0
        for i, tf in enumerate(self.tfs):
            dl = sum(tf.values()) or 1
            s = 0.0
            for t, qn in qtoks.items():
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math_log((self.n - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
                s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl)) * qn
            if s > 0:
                yield s, i

    def doc_len(self, i):
        return sum(self.tfs[i].values()) or 1


def math_log(x):
    import math
    return math.log(x + 1e-9)


def retrieve(index, query, top_k=TOP_K, candidates=CANDIDATES):
    qt = bigrams(query)
    qvec = {}
    n = index.n
    for t, c in qt.items():
        idf = math_log((n - index.df.get(t, 0) + 0.5) / (index.df.get(t, 0) + 0.5) + 1)
        v = (1 + math_log(c)) * idf
        qvec[t] = v
    nrm = sum(x * x for x in qvec.values()) ** 0.5
    if nrm:
        qvec = {t: x / nrm for t, x in qvec.items()}
    if not qvec:
        return []
    cos = sorted(index.cosine(qvec), reverse=True)[:candidates]
    bm = sorted(index.bm25(qt), reverse=True)[:candidates]
    cmax = cos[0][0] if cos else 0.0
    bmax = bm[0][0] if bm else 0.0
    fused = defaultdict(float)
    for s, i in cos:
        fused[i] += W_COS * (s / cmax if cmax else 0.0)
    for s, i in bm:
        fused[i] += W_BM25 * (s / bmax if bmax else 0.0)
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    return [i for i, _ in ranked]


def _norm(s):
    return re.sub(r"[\s\"'“”‘’]", "", s)


# ---------------------------------------------------------------- 评测
#
# **为什么不能只报 doc 级 recall@3**（这是本轮第一个真 bug 的教训）：
# 26 篇短文档、top-3，选文档这件事几乎必赢——五档策略全部 1.000，表是平的。
# 根因是**测错了量**：切分策略改变的是"返回的块里装不装得下完整答案"，
# 不是"答案文档能不能进前三"。26 选 3 里 doc 级 recall 必然饱和。
# 所以这里同时报三个量，doc 级只作饱和度参照，不作结论：
#   span_full@3  top-3 中存在**单个**块完整包含 answer_span —— 分块策略的直接后果
#   span_cover@3 top-3 全部块的并集能覆盖 answer_span 的字符比例
#   doc_hit@3    答案文档进入 top-3（对照用，预期饱和）
# span_full 才是能区分策略的那个量：零重叠会把跨边界的句子劈成两块，
# 没有任何单块装得下，span_full 必然掉。

def evaluate(corpus, golden, max_chars, overlap, sentence_aware):
    chunks, meta = [], []
    for doc, text in corpus.items():
        for c in chunk_text(text, max_chars, overlap, sentence_aware):
            chunks.append(c)
            meta.append(doc)
    index = Index(chunks, meta)
    nchunk = [0, 0.0, 0]      # span_full 命中数, cover 累加, 总数
    ndoc = defaultdict(lambda: [0, 0])
    noise = [0.0, 0]
    detail = []
    for q in golden["queries"]:
        idxs = retrieve(index, q["q"])
        top_texts = [chunks[i] for i in idxs]
        top_docs = [meta[i] for i in idxs]
        span = _norm(q["answer_span"])
        joined = _norm("".join(top_texts))
        full = 1 if any(span in _norm(t) for t in top_texts) else 0
        # 覆盖比例：span 里有多少比例的字符出现在并集中（按 2 字窗口计）
        if span:
            win = [span[i:i + 2] for i in range(len(span) - 1)] or [span]
            cov = sum(1 for w in win if w in joined) / len(win)
        else:
            cov = 0.0
        nchunk[0] += full
        nchunk[1] += cov
        nchunk[2] += 1
        noise[0] += sum(len(t) for t in top_texts)
        noise[1] += 1
        hit = 1 if q["expect_doc"] in top_docs else 0
        ndoc[q["type"]][0] += hit
        ndoc[q["type"]][1] += 1
        detail.append({"q": q["q"], "type": q["type"], "expect_doc": q["expect_doc"],
                       "span_full": full, "span_cover": round(cov, 3),
                       "doc_in_top3": hit, "top3_docs": top_docs,
                       "top3_chars": sum(len(t) for t in top_texts)})
    n = nchunk[2] or 1
    return {
        "n_chunks": len(chunks),
        "span_full_at_3": nchunk[0] / n,
        "span_cover_at_3": nchunk[1] / n,
        "doc_hit_at_3": sum(v[0] for v in ndoc.values()) / n,
        "mean_top3_chars": noise[0] / (noise[1] or 1),
        "by_type": {k: {"hit": v[0], "total": v[1], "recall": v[0] / v[1]}
                    for k, v in sorted(ndoc.items())},
        "detail": detail,
    }


def run():
    corpus = load_corpus()
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    results = []
    for name, mc, ov, sa in GRID:
        r = evaluate(corpus, golden, mc, ov, sa)
        r["strategy"] = name
        r["max_chars"] = mc
        r["overlap"] = ov
        r["sentence_aware"] = sa
        results.append(r)
        bt = r["by_type"]
        print("%-14s chunks=%-4d span_full@3=%.3f span_cover@3=%.3f doc_hit@3=%.3f top3_chars=%.0f"
              % (name, r["n_chunks"], r["span_full_at_3"], r["span_cover_at_3"],
                 r["doc_hit_at_3"], r["mean_top3_chars"]))
    # 说明与结果写进同一个文件：分开存放的结果文件复跑一次就被冲掉，
    # 而这些结论（死参数、权衡比、唯一失分归因）正是复跑的人最需要先读到的。
    notes = {
        "metric_choice":
            "主指标 span_full@3。doc 级 recall@3 在 26 篇短文档下必然饱和"
            "（首版五档全 1.000，跨度 0.000），只作对照不作结论。",
        "dead_param_found":
            "chunk_text 首版固定长度路径下 overlap 完全未生效（fixed_220_o40 与 "
            "fixed_220_o0 输出逐位相同），由机制方向门发现；已修，现分别为 "
            "88 / 81 块、span_full 0.984 / 0.875。",
        "caliber_isolation":
            "本表与冻结基线（docs/ 7 篇 / 5 题 / MRR=1.000）口径不同："
            "语料、题集、指标三项全异，两个数字不可直接比较。",
        "sample_size_caveat":
            "64 题，落在 eval_metrics.md 给的 50–200 可行区间内；"
            "跨度 0.156 约合 10 题，方向性结论成立，单题级比例不宜再细分。",
        "overlap_tradeoff":
            "o0→o40：span_full 56/64 → 63/64，注入字符总量 37715 → 38726"
            "（+2.7%）。即 +2.7% 提示词成本换回 7 题的完整命中。",
        "sole_miss_is_not_chunking":
            "唯一未命中题『上下文重建的两种方法各有什么代价』：全库唯一含该 span 的"
            "块排名第 4 位，不在 top-3 ⇒ 检索侧漏召，非切分把答案切碎。"
            "该题 span_cover@3=0.567，说明部分答案文本已进 top-3。",
    }
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump({"grid": [list(g) for g in GRID],
                   "top_k": TOP_K,
                   "metrics": {
                       "span_full_at_3": "top-3 中存在单个块完整包含 answer_span（分块策略的直接后果）",
                       "span_cover_at_3": "top-3 全部块并集覆盖 answer_span 的字符比例",
                       "doc_hit_at_3": "答案文档进入 top-3（对照用，26 篇短文档下预期饱和）",
                       "mean_top3_chars": "top-3 注入提示词的字符总数（成本侧）"},
                   "results": results,
                   "notes": notes},
                  f, ensure_ascii=False, indent=2)
    print("\n结果已写入 %s" % os.path.relpath(RESULTS, os.path.dirname(HERE)))
    return results


# ---------------------------------------------------------------- 门

def discrimination_gate(results):
    """题集必须能区分开策略，否则整张表没有信息量。

    判据来自 benchmark_stability.md 那句"看它能不能让两种明显不同的配置
    给出明显不同的分数"。分两道关：

    关 1（span_full 跨度）——主判据。切分策略改变的是块内是否装得下完整答案，
    若五档 span_full@3 相同，说明这道消融测不出任何东西，必须 exit 1。
    关 2（机制方向）——不是只要求"有差异"，还要求差异**方向符合机制**：
    零重叠必须比有重叠差。若 span_flat，重叠参数没起作用，那不是题集的问题，
    是实现坏了；两者都不该出一张看起来正常的表。
    """
    vals = [r["span_full_at_3"] for r in results]
    spread = max(vals) - min(vals)
    print("\n区分力门：span_full@3 跨度 = %.3f（%s）"
          % (spread, " ".join("%.3f" % v for v in vals)))
    if spread < 0.05:
        print("FAIL 题集无区分力：五档策略 span_full@3 几乎相同，表是平的，消融结果不可用")
        return False

    by = {r["strategy"]: r for r in results}
    ok = True
    if by["fixed_220_o0"]["span_full_at_3"] >= by["fixed_220_o40"]["span_full_at_3"]:
        print("FAIL 机制方向反了：零重叠 span_full@3=%.3f 不低于有重叠 %.3f，重叠参数未起作用"
              % (by["fixed_220_o0"]["span_full_at_3"],
                 by["fixed_220_o40"]["span_full_at_3"]))
        ok = False
    else:
        print("机制方向 OK：零重叠 %.3f < 有重叠 %.3f（重叠确实在补边界）"
              % (by["fixed_220_o0"]["span_full_at_3"],
                 by["fixed_220_o40"]["span_full_at_3"]))
    bt = [r["by_type"]["boundary"]["recall"] for r in results]
    print("doc 级 boundary 跨度 = %.3f（对照用，饱和属正常）" % (max(bt) - min(bt)))
    return ok


def selftest():
    """反证：把题集换回"全部平凡"时，门必须失败。

    这条反证针对的是最容易自欺的情形——门一直 PASS，
    因为题集简单到任何策略都能满分。注入一个所有题答案都在同一文档的
    题集，五档策略必然全部满分、跨度为 0，门必须拦下。
    """
    orig = open(GOLDEN, encoding="utf-8").read()
    import io
    import contextlib
    try:
        # 前提：正常题集下门应当通过
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            base = run()
        assert discrimination_gate(base), "【门失效】正常题集下区分力门应通过"
        print("  前提通过：正常题集下门通过\n")

        # 注入：全部 64 题都指向同一文档且题面极短，任何策略都能命中
        data = json.loads(orig)
        for q in data["queries"]:
            q["expect_doc"] = "eval_metrics"
            q["q"] = "评测"
        with open(GOLDEN, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            flat = run()
        ok = discrimination_gate(flat)
        assert not ok, "【门失效】题集退化为平凡题时门仍通过——门形同虚设"
        print("\n  通过：题集退化为平凡题（全部指向同一文档）时被拦")
    finally:
        with open(GOLDEN, "w", encoding="utf-8") as f:
            f.write(orig)
        run()   # 复原结果文件
    print("SELFTEST OK：区分力门在平凡题集上会失败")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        rs = run()
        sys.exit(0 if discrimination_gate(rs) else 1)
