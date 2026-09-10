#!/usr/bin/env python3
"""W3-P3C：neardup与N-swap真信号升级为golden级自然问法（单答案复核+改写）。

只读基线（rag_demo.py + docs/ 不碰）；35 chunk扩展池冻结；7格点口径与P3B一致。
候选 production：自然用户问法，无整chunk粘贴、无词频堆叠、无全毁式字符交换。
每候选单答案复核（expect_doc/expect_chunks + label_note + review记录）后跑7格点，
相对 ref(w_cos=0.6) 计 hit/rank 分歧。失败候选如实保留（判负），不删除。
用法：PYTHONDONTWRITEBYTECODE=1 python3 rag_min_demo/bline/w3_p3c_nat.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rag_demo
from w1_hybrid import rrf_fusion, weighted_fusion
from w3_harness import build_extended, norm_item, strategies

with open(os.path.join(HERE, "w1_config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
chunks0, index0 = build_extended()
assert len(chunks0) == 35, len(chunks0)
sts0, names = strategies(cfg)
assert len(sts0) == 7, len(sts0)
REF = "w_cos=0.6"


def score_with(st, query, top_k):
    qtoks = rag_demo.bigrams(query)
    qvec = index0._vec(qtoks)
    if not any(qvec.values()):
        return []
    cos = {i: s for s, i in index0.cosine(qvec)}
    bm = {i: s for s, i in index0.bm25(qtoks)}
    if st[0] == "weighted":
        ranked = weighted_fusion(cos, bm, w_cos=st[1],
                                 candidates=st[2], top_k=top_k)
    else:
        ranked = rrf_fusion(cos, bm, k=st[1],
                            candidates=st[2], top_k=top_k)
    return [(s, chunks0[i]) for s, i in ranked]


# ---------------- Round1 自然候选（人工改写，provenance注明来源信号） ----------------
CANDS = [
    dict(cid="N1-milvus-nat", src="neardup/Milvus小数据量",
         q="我们现在数据量也就几万条，运维人手也不多，有必要上Milvus这种向量数据库服务吗，还是单机FAISS先顶着？",
         expect_doc="vector_db", expect_chunks=["vector_db#c0"],
         label_note="单机FAISS先单机后服务句在c0；问法自然化自seed原问，未粘贴原文。",
         intent="保易混：点名Milvus/FAISS/单机，测BM25是否被服务名带偏、cos能否守住c0"),
    dict(cid="N2-select-nat", src="neardup/向量数据库选型",
         q="向量数据库选型和调优一般抓哪几个旋钮？HNSW的efsearch和混合检索具体要怎么定？",
         expect_doc="vector_db", expect_chunks=["vector_db#c0", "vector_db#c1"],
         label_note="整文档级问题沿用seed口径：选型c0、调优c1皆相关；关键词为自然追问用语。",
         intent="保易混：旋钮/HNSW/混合检索同时命中c1，测两路是否分化"),
    dict(cid="N3-chunk-nat", src="N-swap/检索到了也答不全",
         q="切分做完检索分数看着挺高，但答案总是不全，像是前后指代丢了，该换哪种分块策略？",
         expect_doc="chunking", expect_chunks=["chunking#c0"],
         label_note="同义改写seed原问（检索到了也答不全/前后指代丢失/分块策略皆在c0），未用原句。",
         intent="N-swap自然化：1处同义改写+口语化，测自然噪声下两路是否分叉"),
    dict(cid="N4-eval-nat", src="neardup/新老版本对比",
         q="线上迭代了个新版本，老板问提升了多少，我直接报新版本的绝对分数行吗，还是必须拿同一批问题做AB对比只看差值？",
         expect_doc="eval", expect_chunks=["eval#c1"],
         label_note="只看差值不看绝对值句在c1；问法为自然工作场景追问。",
         intent="保易混：绝对分数/AB对比/差值三词同现，测BM25是否被带偏"),
    dict(cid="N5-steps-nat", src="N-swap/Agent最大步数",
         q="Agent循环跑起来有时一直打转停不下来还烧钱，工程上一般怎么防？",
         expect_doc="agent_tools", expect_chunks=["agent_tools#c0"],
         label_note="最大步数防空转句在c0；改写去掉关键词最大步数，纯语义指向。",
         intent="N-swap自然化：去关键词纯语义问法，测cos能否独立命中"),
    dict(cid="N6-tools-nat", src="N-swap/Agent查知识实现",
         q="Agent缺背景知识的时候，怎么让它自己去查资料，这个检索工具是怎么接进循环的？",
         expect_doc="agent_tools", expect_chunks=["agent_tools#c0"],
         label_note="检索当工具句在c0；轻度改写，保留检索/工具关键词。",
         intent="N-swap自然化：保留关键词的轻度改写，对照N5去关键词版"),
    dict(cid="N7-travel-nat", src="N-swap/第二天的行程（反向题）",
         q="第二天去哪玩，有什么推荐？",
         expect_doc="travel", expect_chunks=["travel#c0"],
         label_note="单chunk文档；反向题的自然缩短版（原N-trunc机械截半的对照）。",
         intent="反向题自然化：短问下加权rank3/rrf-miss模式是否保持"),
]

# ---------------- Round2 边界变体（首轮全rank1后的有界加压，仍自然+单答案） ----------------
ROUND2 = [
    dict(cid="R2-1-milvus-eval", src="neardup/Milvus小数据量+R2",
         q="几万条数据量，单机FAISS还是上Milvus服务？我们想先拿评测数据对比一下再拍板。",
         expect_doc="vector_db", expect_chunks=["vector_db#c0"],
         label_note="问句是选型（c0），后半句评测数据为自然补充的跨主题从句（eval词汇），单答案维持选型。",
         intent="跨主题从句施压：BM25是否被评测词汇拉向eval、cos能否守住vector_db"),
    dict(cid="R2-2-eval-frozen", src="neardup/新老版本对比+R2",
         q="新老版本对比只看差值不看绝对分数的话，评测集是不是必须冻结？",
         expect_doc="eval", expect_chunks=["eval#c1"],
         label_note="复核修正：差值句与冻结纪律同在c1（c0为golden集/recall/忠实度），故单标c1；双遍复核已记。",
         intent="单chunk多纪律点：自然问法下两路是否仍一致"),
    dict(cid="R2-3-chunk-tradeoff", src="N-swap/检索到了也答不全+R2",
         q="分块太大把噪声带进提示词浪费窗口，分块太小切断语义答不全，滑动窗口保留重叠能两全吗？",
         expect_doc="chunking", expect_chunks=["chunking#c0"],
         label_note="大/小/重叠三现象全在c0；用金chunk自身词汇重述，测自然上限（预期全一致，阴性对照）。",
         intent="阴性对照：金词汇重述应全格点一致，验证口径不误报"),
    dict(cid="R2-4-agent-guard", src="N-swap/Agent最大步数+R2",
         q="Agent循环空转烧钱，除了限制最大步数，工具返回太长挤爆上下文要一起截断吗？",
         expect_doc="agent_tools", expect_chunks=["agent_tools#c0"],
         label_note="单chunk文档；最大步数+截断两工程点同在c0，单答案。",
         intent="同chunk多工程点：自然问法下两路是否仍一致"),
]

CANDS = CANDS + ROUND2

rows = []
for c in CANDS:
    doc, chs = c["expect_doc"], c["expect_chunks"]
    rec_d, rec_c = {}, {}
    for st, name in zip(sts0, names):
        res = score_with(st, c["q"], cfg["top_k"])
        d = next((i + 1 for i, (_, cc) in enumerate(res)
                  if cc["doc_id"] == doc), None)
        kk = next((i + 1 for i, (_, cc) in enumerate(res)
                   if cc["chunk_id"] in chs), None)
        rec_d[name] = d
        rec_c[name] = kk
    ref_d = rec_d[REF]
    hd = sum(1 for v in rec_d.values() if (v is None) != (ref_d is None))
    rd = len(set(str(v) for v in rec_d.values())) > 1
    rows.append({"cid": c["cid"], "src": c["src"], "q": c["q"],
                 "expect_doc": doc, "expect_chunks": chs,
                 "label_note": c["label_note"], "intent": c["intent"],
                 "doc_ranks": rec_d, "chunk_ranks": rec_c,
                 "hit_div": hd, "rank_div": rd})

print("=== P3C-Round1 自然候选 × 7格点 ===")
for r in rows:
    print(f"[{r['cid']}] hit_div={r['hit_div']} rank_div={r['rank_div']} doc={r['doc_ranks']}")
    print(f"   Q: {r['q'][:60]}…")

out = {"n": len(rows), "hitdiv": sum(1 for r in rows if r["hit_div"]),
       "rankdiv": sum(1 for r in rows if r["rank_div"]), "rows": rows,
       "pool_chunks": 35, "ref": REF,
       "top_k": cfg["top_k"], "candidates": cfg["candidates"]}
p = os.path.join(HERE, "w3_p3c_nat_results.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"[P3C] 结果已落盘 {p}")
