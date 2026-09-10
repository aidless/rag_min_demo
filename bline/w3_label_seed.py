#!/usr/bin/env python3
"""W3-P2：由 w2_golden.json 派生 w3_golden.json（32题seed，双标注）。

只做加字段，不改题面：expect_doc = 原 expect；
expect_chunks = 人工逐题阅读chunk原文后标注（单chunk文档取唯一chunk，
多chunk文档按答案所在chunk标注，整文档级问题标两chunk并附理由）。
用法：PYTHONDONTWRITEBYTECODE=1 python3 rag_min_demo/bline/w3_label_seed.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# idx -> (expect_chunks, label_note)
LABELS = {
    0: (["rag_overview#c0"], "单chunk文档"),
    1: (["rag_overview#c0"], "单chunk文档；工作流句在c0"),
    2: (["rag_overview#c0"], "单chunk文档；免微调优势句在c0"),
    3: (["rag_overview#c0"], "单chunk文档；拼进提示词属工作流程"),
    4: (["rag_overview#c0"], "单chunk文档；端到端正确率仅c0出现"),
    5: (["chunking#c0"], "太大/太小两句皆在c0"),
    6: (["chunking#c0"], "固定长度缺点句在c0"),
    7: (["chunking#c0"], "缓解边界截断句在c0"),
    8: (["chunking#c1"], "无银弹实测句仅c1，c0无此句"),
    9: (["chunking#c0"], "检索到了也答不全原句在c0"),
    10: (["vector_db#c0", "vector_db#c1"], "整文档级问题：选型在c0、调优在c1，两chunk皆相关"),
    11: (["vector_db#c0"], "单机FAISS先单机后服务句在c0"),
    12: (["vector_db#c1"], "efsearch精度延迟tradeoff句在c1"),
    13: (["vector_db#c1"], "纯向量翻车句在c1"),
    14: (["vector_db#c0", "vector_db#c1"], "L2归一化句因40字重叠同时出现在两chunk"),
    15: (["eval#c0", "eval#c1"], "整文档级问题：检索/生成侧在c0，评测纪律在c1"),
    16: (["eval#c0"], "golden集定义句在c0"),
    17: (["eval#c1"], "改动重跑全量评测句在c1"),
    18: (["eval#c1"], "只看差值句在c1"),
    19: (["eval#c0"], "断言找依据完整句在c0；c1仅截断重复片段，不计"),
    20: (["agent_tools#c0"], "单chunk文档"),
    21: (["agent_tools#c0"], "单chunk文档；最大步数防空转句在c0"),
    22: (["agent_tools#c0"], "单chunk文档；截断防挤爆句在c0"),
    23: (["agent_tools#c0"], "单chunk文档；检索当工具句在c0"),
    24: (["cooking#c0"], "单chunk文档"),
    25: (["cooking#c0"], "单chunk文档"),
    26: (["cooking#c0"], "单chunk文档"),
    27: (["cooking#c0"], "单chunk文档"),
    28: (["travel#c0"], "单chunk文档"),
    29: (["travel#c0"], "单chunk文档"),
    30: (["travel#c0"], "单chunk文档"),
    31: (["travel#c0"], "单chunk文档"),
}


def main():
    with open(os.path.join(HERE, "w2_golden.json"), encoding="utf-8") as f:
        w2 = json.load(f)
    queries = w2["queries"]
    assert len(queries) == 32, f"W2题数漂移：{len(queries)}"
    assert set(LABELS) == set(range(32)), "标注表与题数不对齐"
    out = []
    for i, q in enumerate(queries):
        chunks, note = LABELS[i]
        out.append({"q": q["q"], "type": q.get("type"), "note": q.get("note"),
                    "expect_doc": q["expect"],
                    "expect_chunks": chunks, "label_note": note})
    golden = {"version": "w3-seed32",
              "desc": "W2 32题SILVER双标注seed；P3再追加≥10对抗题",
              "expected_doc_field": "expect_doc",
              "expected_chunk_field": "expect_chunks（命中任一即chunk级命中）",
              "queries": out}
    p = os.path.join(HERE, "w3_golden.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)
    print(f"[W3-P2] w3_golden.json 已落盘 {p}（{len(out)}题）")


if __name__ == "__main__":
    main()
