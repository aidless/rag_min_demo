#!/usr/bin/env python3
"""W3-P3 合成 v3 golden：seed32（标签一字不动）+ adv12，并自验证。
自验证内容：adv数量=12、seed数量=32且标签与w3_golden.json一致、
每题ref(w_cos=0.6)排名与w3_adversarial.json记录一致、7格点分歧率重算。
只读基线；PYTHONDONTWRITEBYTECODE=1 运行。
用法：PYTHONDONTWRITEBYTECODE=1 python3 rag_min_demo/bline/w3_build_v3.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from w3_harness import build_extended, eval_strategy, strategies  # 只读复用

with open(os.path.join(HERE, "w1_config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
with open(os.path.join(HERE, "w3_golden.json"), encoding="utf-8") as f:
    seed = json.load(f)
with open(os.path.join(HERE, "w3_adversarial.json"), encoding="utf-8") as f:
    adv = json.load(f)

assert seed["version"] == "w3-seed32", seed["version"]
assert len(seed["queries"]) == 32, len(seed["queries"])
assert adv["version"] == "w3-adv12", adv["version"]
assert len(adv["adversarial"]) == 12, len(adv["adversarial"])

chunks, index = build_extended()
assert len(chunks) == 35, len(chunks)
sts, names = strategies(cfg)
ref = ("weighted", 0.6, cfg["candidates"])
assert ref in sts, "ref策略不在网格内"

n_div = 0
for item in adv["adversarial"]:
    assert item["expect_doc"] and item["expect_chunks"], item["q"]
    ranks_d, ranks_c = [], []
    for st in sts:
        r = eval_strategy(st, chunks, index, [item], cfg["top_k"])
        ranks_d.append(r["ranks"][0]["doc_rank"])
        ranks_c.append(r["ranks"][0]["chunk_rank"])
    div = (len(set(map(str, ranks_d))) > 1
           or len(set(map(str, ranks_c))) > 1)
    n_div += div
    m = item["measured"]
    assert ranks_d[names.index("w_cos=0.6")] == m["ref_doc_rank"], \
        (item["q"], ranks_d, m["ref_doc_rank"])
    assert ranks_c[names.index("w_cos=0.6")] == m["ref_chunk_rank"], \
        (item["q"], ranks_c, m["ref_chunk_rank"])
    assert m["divergence_7grid_doc"] == 0 and not div, item["q"]
    print(f"  OK {item['q'][:24]:26s} ref_doc={m['ref_doc_rank']} "
          f"ref_chunk={m['ref_chunk_rank']} 分歧=0")

v3 = {
    "version": "w3-v3-44",
    "desc": "seed32标签冻结 + adv12；P4全网格实测集",
    "changelog": [
        "seed32一题未改（含Q14长短文本与翻车题原标签，保持连续性）",
        "新增adv12（5轮56探针精选，全部单答案，7格点零分歧）",
        "A12采用同文档双标（40字重叠区规则，见L-blinew3-02）",
    ],
    "drift_flags": [
        "seed-Q14长短文本：embedding_select#c0/c1同样完整回答（P1扩语料引入），"
        "extended池下cos路top1已是embedding#c1；标签维持vector_db，P4须做剔除敏感性分析",
        "seed-翻车题：hybrid_retrieval#c0同样完整回答，双路margin仅1.05；"
        "标签维持vector_db#c1，P4须做剔除敏感性分析",
        "seed-Q16 golden集：extended池下双路top1均为recall_debug#c1（P2已报干扰），标签维持eval#c0",
    ],
    "expected_doc_field": "expect_doc",
    "expected_chunk_field": "expect_chunks",
    "queries": seed["queries"] + [
        {k: item[k] for k in ("q", "type", "expect_doc", "expect_chunks")
         } | {"label_note": item["construct"]["direction"] + "；"
                              + item["construct"]["note"],
              "trap": {"doc": item["construct"]["trap_doc"],
                       "chunks": item["construct"]["trap_chunks"]}}
        for item in adv["adversarial"]
    ],
}
assert len(v3["queries"]) == 44
p = os.path.join(HERE, "w3_golden_v3.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(v3, f, ensure_ascii=False, indent=2)
print(f"[W3-P3] 对抗子集分歧率 {n_div}/12；v3-44已落盘 {p}")
sys.exit(0 if n_div == 0 else 1)
