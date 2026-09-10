#!/usr/bin/env python3
"""W3-P3B剩余三方向挖掘（全做）：depth / noise / near_dup。

诚实设计（只读基线，不污染v3 golden；结论只用于"权重是否有信号"）：
- D-depth：不动问法，只变候选池深度 candidates={5,10,20,35}（C10为对照，
  P4已知零分歧）与 top_k={1,5}（candidates=10）。比较口径为“同深度同top_k
  下7格点相对ref(w_cos=0.6)是否分歧”。
- N-noise：纯机械变换，不编造任何文本：
    N-del10  每第10个字符删1个；N-swap  相邻字符两两交换；N-trunc  只留前一半。
- X-neardup：在内存中追加3个合成干扰chunk（金chunk原文分别在1/4、1/2、3/4
  处删1字符，doc_id="__interf__"，不写盘），池35→38后重建索引再跑7格点。
  合成干扰如实披露，是机制探针，不进golden。
用法：PYTHONDONTWRITEBYTECODE=1 python3 rag_min_demo/bline/w3_p3b_rest.py
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
with open(os.path.join(HERE, "w3_golden_v3.json"), encoding="utf-8") as f:
    v3 = json.load(f)
assert v3["version"] == "w3-v3-44", v3["version"]
seeds = [q for q in v3["queries"] if "trap" not in q]
assert len(seeds) == 32, len(seeds)

chunks0, index0 = build_extended()
assert len(chunks0) == 35, len(chunks0)
sts0, names = strategies(cfg)
assert len(sts0) == 7, len(sts0)
REF = "w_cos=0.6"


def score_with(st, chunks, index, query, top_k):
    qtoks = rag_demo.bigrams(query)
    qvec = index._vec(qtoks)
    if not any(qvec.values()):
        return []
    cos = {i: s for s, i in index.cosine(qvec)}
    bm = {i: s for s, i in index.bm25(qtoks)}
    if st[0] == "weighted":
        ranked = weighted_fusion(cos, bm, w_cos=st[1],
                                 candidates=st[2], top_k=top_k)
    else:
        ranked = rrf_fusion(cos, bm, k=st[1],
                            candidates=st[2], top_k=top_k)
    return [(s, chunks[i]) for s, i in ranked]


def grid_at(cands, chunks, index, qtext, doc, chs, top_k):
    sts = [(k, v, cands) if k == "weighted" else (k, v, cands)
           for k, v in [(s[0], s[1]) for s in sts0]]
    rec = {"doc_rank": {}, "chunk_rank": {}}
    for st, name in zip(sts, names):
        res = score_with(st, chunks, index, qtext, top_k)
        d = next((i + 1 for i, (_, c) in enumerate(res)
                  if c["doc_id"] == doc), None)
        kk = next((i + 1 for i, (_, c) in enumerate(res)
                   if chs and c["chunk_id"] in chs), None)
        rec["doc_rank"][name] = d
        rec["chunk_rank"][name] = kk
    ref_d = rec["doc_rank"][REF]
    hit_div = sum(1 for v in rec["doc_rank"].values()
                  if (v is None) != (ref_d is None))
    rank_div = len(set(str(v) for v in rec["doc_rank"].values())) > 1
    return rec, hit_div, rank_div


# ---------------- D-depth ----------------
depth_rows = []
for item in seeds:
    doc, chs = norm_item(item)
    for cands in (5, 10, 20, 35):
        rec, hd, rd = grid_at(cands, chunks0, index0, item["q"], doc, chs,
                              cfg["top_k"])
        depth_rows.append({"seed_q": item["q"], "variant": f"C{cands}",
                           "top_k": cfg["top_k"], "doc_ranks": rec["doc_rank"],
                           "chunk_ranks": rec["chunk_rank"],
                           "hit_div": hd, "rank_div": rd})
    for tk in (1, 5):
        rec, hd, rd = grid_at(cfg["candidates"], chunks0, index0, item["q"],
                              doc, chs, tk)
        depth_rows.append({"seed_q": item["q"], "variant": f"T{tk}",
                           "top_k": tk, "doc_ranks": rec["doc_rank"],
                           "chunk_ranks": rec["chunk_rank"],
                           "hit_div": hd, "rank_div": rd})

# ---------------- N-noise（纯机械变换） ----------------
def n_del10(q):
    return "".join(ch for i, ch in enumerate(q) if (i + 1) % 10 != 0)

def n_swap(q):
    a = list(q)
    for i in range(0, len(a) - 1, 2):
        a[i], a[i + 1] = a[i + 1], a[i]
    return "".join(a)

def n_trunc(q):
    return q[:len(q) // 2]

noise_rows = []
for item in seeds:
    doc, chs = norm_item(item)
    for vname, fn in (("N-del10", n_del10), ("N-swap", n_swap),
                      ("N-trunc", n_trunc)):
        qtext = fn(item["q"])
        rec, hd, rd = grid_at(cfg["candidates"], chunks0, index0, qtext,
                              doc, chs, cfg["top_k"])
        noise_rows.append({"seed_q": item["q"], "variant": vname,
                           "qtext": qtext, "doc_ranks": rec["doc_rank"],
                           "chunk_ranks": rec["chunk_rank"],
                           "hit_div": hd, "rank_div": rd})

# ---------------- X-neardup（内存合成干扰，不写盘） ----------------
by_id = {c["chunk_id"]: c for c in chunks0}
near_rows = []
for item in seeds:
    doc, chs = norm_item(item)
    if not chs:
        continue
    gold = by_id[chs[0]]["text"]
    n = len(gold)
    cuts = sorted(set([n // 4, n // 2, 3 * n // 4]))
    synth = [{"chunk_id": f"__interf__#s{k}", "doc_id": "__interf__",
              "title": "__interf__", "text": gold[:p] + gold[p + 1:],
              "chars": len(gold) - 1}
             for k, p in enumerate(cuts) if 0 < p < n]
    chunks = chunks0 + synth
    toks = [rag_demo.bigrams(c["text"]) for c in chunks]
    rag_demo.chunk_tokens_global = toks
    index = rag_demo.TfidfIndex(toks)
    assert len(chunks) == 35 + len(synth)
    rec, hd, rd = grid_at(cfg["candidates"], chunks, index, item["q"],
                          doc, chs, cfg["top_k"])
    # 金chunk vs 合成干扰：ref首位是否被合成抢占（机制信号，非命中口径）
    res_ref = score_with(("weighted", 0.6, cfg["candidates"]), chunks, index,
                         item["q"], cfg["top_k"])
    top1 = res_ref[0][1]["chunk_id"] if res_ref else None
    near_rows.append({"seed_q": item["q"], "variant": "X-neardup3",
                       "n_synth": len(synth), "ref_top1": top1,
                       "ref_top1_is_synth": bool(top1 and
                                                 top1.startswith("__interf__")),
                       "doc_ranks": rec["doc_rank"],
                       "chunk_ranks": rec["chunk_rank"],
                       "hit_div": hd, "rank_div": rd})
rag_demo.chunk_tokens_global = [rag_demo.bigrams(c["text"]) for c in chunks0]

def summ(rows):
    return {"n": len(rows),
            "hitdiv": sum(1 for r in rows if r["hit_div"]),
            "rankdiv": sum(1 for r in rows if r["rank_div"])}

out = {"depth": summ(depth_rows), "noise": summ(noise_rows),
       "neardup": summ(near_rows),
       "depth_rows": depth_rows, "noise_rows": noise_rows,
       "near_rows": near_rows}
print(f"[P3B-rest] depth {out['depth']} | noise {out['noise']} "
      f"| neardup {out['neardup']}")
for tag, rows in (("depth", depth_rows), ("noise", noise_rows),
                  ("neardup", near_rows)):
    print(f"--- {tag} hit分歧行 ---")
    for r in rows:
        if r["hit_div"]:
            print(f"[{r['variant']}] {r['seed_q'][:40]}… doc={r['doc_ranks']}")
    print(f"--- {tag} rank分歧行（hit同，前10） ---")
    shown = 0
    for r in rows:
        if not r["hit_div"] and r["rank_div"] and shown < 10:
            print(f"[{r['variant']}] {r['seed_q'][:40]}… doc={r['doc_ranks']}")
            shown += 1
print(f"[P3B-rest] neardup合成抢占ref首位 "
      f"{sum(1 for r in near_rows if r['ref_top1_is_synth'])}/{len(near_rows)}")

p = os.path.join(HERE, "w3_p3b_rest_results.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"[P3B-rest] 结果已落盘 {p}")
