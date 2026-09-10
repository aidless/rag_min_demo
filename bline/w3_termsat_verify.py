#!/usr/bin/env python3
"""P3B-termsat复核：hit分歧独立重算 + 重复次数扫描 + 两路归因。

复核点：
1. 全部hit分歧行按构造参数重构查询重跑7格点，与w3_termsat_results.json逐格一致。
2. 首个hit用例做重复次数扫描（rep×1..×12）：分歧随堆叠强度单调出现
   （可控信号）还是孤立抖动（fluke）。
3. 两路归因：比较短问 vs 堆叠查询下纯cos/BM25分数向量变化——BM25的query词频
   是否真盲（qc未参与），cos路是否被log-tf+L2稀释主导；同时量化T-rep引入的
   边界二元组新类型对BM25的附带扰动（诚实披露非对称性）。
只读基线与已有文件；输出复核JSON。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import rag_demo
from w3_harness import build_extended, norm_item, score_for, strategies

with open(os.path.join(HERE, "w1_config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
with open(os.path.join(HERE, "w3_golden_v3.json"), encoding="utf-8") as f:
    v3 = json.load(f)
seeds = [q for q in v3["queries"] if "trap" not in q]
with open(os.path.join(HERE, "w3_termsat_results.json"), encoding="utf-8") as f:
    prev = json.load(f)

chunks, index = build_extended()
sts, names = strategies(cfg)
REF = "w_cos=0.6"


def full_ranks(qtext, doc, chs):
    out = {}
    for st, name in zip(sts, names):
        res = score_for(st, chunks, index, qtext, cfg["top_k"])
        d = next((i + 1 for i, (_, c) in enumerate(res)
                  if c["doc_id"] == doc), None)
        out[name] = d
    return out


def rep_of(variant, R):
    n = {"T-rep3": 3, "T-rep8": 8}[variant]
    return R * n


# 1. hit行逐格重算
HITS = [r for r in prev["rows"] if r["hit_div_doc_vs_ref"] > 0]
print(f"[复核] hit行数={len(HITS)}")
rerun_ok = True
for h in HITS:
    item = next(q for q in seeds if q["q"] == h["seed_q"])
    doc, chs = norm_item(item)
    qtext = item["q"] + "\n" + rep_of(h["variant"], h["rare_bigram"])
    assert len(qtext) == h["qlen"], (len(qtext), h["qlen"])
    rr = full_ranks(qtext, doc, chs)
    ok = rr == h["doc_ranks"]
    rerun_ok &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] [{h['variant']}] "
          f"{h['seed_q'][:30]}… rare={h['rare_bigram']}")
assert rerun_ok, "确定性重算不一致"
print("  [PASS] 全部hit行7格点逐格重算一致（确定性）")

# 2/3. 首个hit用例：重复扫描 + 两路归因
H0 = HITS[0]
item0 = next(q for q in seeds if q["q"] == H0["seed_q"])
doc0, chs0 = norm_item(item0)
R0 = H0["rare_bigram"]
print(f"[扫描] seed={item0['q'][:40]}… rare={R0} expect={doc0}/{chs0}")


def pure_vecs(qtext):
    qtoks = rag_demo.bigrams(qtext)
    qvec = index._vec(qtoks)
    return ({i: s for s, i in index.cosine(qvec)},
            {i: s for s, i in index.bm25(qtoks)})


cos_s, bm_s = pure_vecs(item0["q"])
print(f"  短问纯cos top1={chunks[max(cos_s, key=cos_s.get)]['chunk_id']} "
      f"纯BM25 top1={chunks[max(bm_s, key=bm_s.get)]['chunk_id']}")
scan = {}
for n in (1, 2, 3, 5, 8, 12):
    qt = item0["q"] + "\n" + R0 * n
    cos, bm = pure_vecs(qt)
    # BM25向量相对短问的变化（query词频盲检验：重复不应改变既有类型的BM25分量）
    bmd = max(abs(bm[i] - bm_s[i]) for i in bm)
    cosd = max(abs(cos[i] - cos_s[i]) for i in cos)
    rr = full_ranks(qt, doc0, chs0)
    vals = sorted(set(str(v) for v in rr.values()))
    scan[n] = {"doc_ranks": rr, "max_abs_d_cos": round(cosd, 4),
               "max_abs_d_bm25": round(bmd, 4)}
    print(f"    rep×{n:2d}: {rr} 分歧值集={vals} "
          f"cos漂移={cosd:.4f} bm25漂移={bmd:.4f}")

qt8 = item0["q"] + "\n" + R0 * 8
cos8, bm8 = pure_vecs(qt8)
tc = sorted(cos8, key=cos8.get, reverse=True)[:5]
tb = sorted(bm8, key=bm8.get, reverse=True)[:5]
print("  rep×8纯cos top5:", [(chunks[i]["chunk_id"], round(cos8[i], 4))
                             for i in tc])
print("  rep×8纯BM25 top5:", [(chunks[i]["chunk_id"], round(bm8[i], 4))
                              for i in tb])
# 新增边界二元组类型清单（T-rep附带扰动披露）
t0 = set(rag_demo.bigrams(item0["q"]))
t8 = set(rag_demo.bigrams(qt8))
print(f"  rep引入的新二元组类型: {sorted(t8 - t0)}")

out = {"n_hit_rows": len(HITS), "deterministic_rerun_match": True,
       "scan_seed_q": H0["seed_q"], "rare_bigram": R0,
       "rep_scan": {str(k): v for k, v in scan.items()},
       "rep8_pure_cos_top5": [(chunks[i]["chunk_id"], cos8[i]) for i in tc],
       "rep8_pure_bm25_top5": [(chunks[i]["chunk_id"], bm8[i]) for i in tb],
       "new_types_by_rep": sorted(t8 - t0)}
p = os.path.join(HERE, "w3_termsat_verify.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"[P3B-termsat复核] 已落盘 {p}")
