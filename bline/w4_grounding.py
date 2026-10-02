#!/usr/bin/env python3
"""W4 题集接地门：验证每题 answer_span 逐字存在于 expect_doc 中。

**为什么这道门是 fail-closed 的**：出题时我凭对自己写的语料的记忆填 label，
记忆最擅长的事就是"编出读起来完全合理的答案"。如果 answer_span 实际不在
该文档里，这题要么标错了文档，要么答案是我凭印象写的——两种情况都会让整张
消融表失去意义，而且从数字上看不出来（该文档可能仍然排第一，只是因为它
恰好也含相似词）。

所以：**任何一题接地失败即 exit 1**，不打印报告、不产出分数。
先证明题是对的，再让题去考系统。

用法：python3 bline/w4_grounding.py [--selftest]
退出码：0=全部接地 / 1=存在未接地 / 2=输入缺失
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "w4_corpus")
GOLDEN = os.path.join(HERE, "w4_golden.json")

# 题集允许的类型；出现表外类型即失败（防止悄悄加一类却没纳入统计）
TYPES = {"single", "multi_hop", "boundary"}


def load_corpus():
    out = {}
    for fn in sorted(os.listdir(CORPUS)):
        if fn.endswith((".md", ".txt")):
            with open(os.path.join(CORPUS, fn), encoding="utf-8") as f:
                out[os.path.splitext(fn)[0]] = f.read()
    return out


def normalize(s):
    """比对用归一：抹掉所有空白与常见标点差异，只留实义字符。

    语料是散文，answer_span 是从里面摘的，可能在换行/空格/全角标点上
    有出入。逐字比对会因这些无意义的差异误报，那会让这道门变成噪声。
    但**不能**归一得太狠——去掉标点会让"不支持和反驳"这类题失去区分力，
    所以这里只抹空白和引号，保留标点。
    """
    return re.sub(r"[\s\"'“”‘’]", "", s)


def check():
    if not os.path.isdir(CORPUS):
        print("FATAL 语料目录不存在: %s" % CORPUS, file=sys.stderr)
        return 2
    if not os.path.exists(GOLDEN):
        print("FATAL 题集不存在: %s" % GOLDEN, file=sys.stderr)
        return 2
    corpus = load_corpus()
    with open(GOLDEN, encoding="utf-8") as f:
        data = json.load(f)
    qs = data.get("queries") or []
    if not qs:
        print("FATAL 题集为空", file=sys.stderr)
        return 2

    fails, by_type, seen_q = [], {}, {}
    for i, q in enumerate(qs):
        tag = "#%d %s" % (i + 1, q.get("q", "<无题面>"))
        for field in ("q", "type", "expect_doc", "answer_span"):
            if not q.get(field):
                fails.append((tag, "缺字段 %s" % field))
        if q.get("q") in seen_q:
            fails.append((tag, "题面重复（与 #%d 相同）" % seen_q[q["q"]]))
        else:
            seen_q[q["q"]] = i + 1
        t = q.get("type")
        if t and t not in TYPES:
            fails.append((tag, "未知 type=%r（允许 %s）" % (t, sorted(TYPES))))
        by_type[t] = by_type.get(t, 0) + 1

        doc = q.get("expect_doc")
        if doc and doc not in corpus:
            fails.append((tag, "expect_doc=%r 不在语料中" % doc))
            continue
        span = normalize(q.get("answer_span", ""))
        if span and doc in corpus and span not in normalize(corpus[doc]):
            # 定位它到底在哪个文档里——出题人需要这个才能改，而不是只知道"错了"
            where = [d for d, txt in corpus.items() if span in normalize(txt)]
            fails.append((tag, "answer_span 不在 %s 中%s"
                          % (doc, ("，实际出现在: " + ",".join(where)) if where
                             else "，且不在任何文档中（疑似凭印象编造）")))

    print("题数 %d | 语料 %d 篇" % (len(qs), len(corpus)))
    print("类型分布 " + " | ".join("%s=%d" % (k, by_type.get(k, 0))
                                   for k in sorted(TYPES)))
    if fails:
        print("\n接地失败 %d 条：" % len(fails))
        for tag, why in fails:
            print("  FAIL %-58s %s" % (tag[:58], why))
        return 1
    print("全部接地 OK：每题 answer_span 均逐字存在于 expect_doc")
    return 0


def selftest():
    """反证：注入违规必须 exit 1（AGENTS.md 第 7 条：门必须能失败）。

    两条容易被忽略的前提，缺了这条 selftest 就是自欺：
    1. 注入前先验基线干净——否则 rc==1 可能是题集本来就脏，四条断言全变空断言；
    2. 断言**失败条数恰为 1 且指名被注入的字段**——只断言 rc==1 的话，
       门"永远失败"也算通过。
    """
    cases = [
        ("answer_span 落在别的文档（标签错）", lambda q: q.update(
            answer_span="分块参数没有银弹，必须拿检索召回率实测"), "answer_span"),
        ("answer_span 根本不存在（编造）", lambda q: q.update(
            answer_span="这一句在任何文档里都没有出现过用于反证"), "answer_span"),
        ("expect_doc 指向不存在的文档", lambda q: q.update(
            expect_doc="no_such_doc"), "expect_doc"),
        ("未知 type（悄悄加一类却没纳入统计）", lambda q: q.update(
            type="brand_new_type"), "type"),
    ]
    orig = open(GOLDEN, encoding="utf-8").read()
    # 前提 1：基线必须先干净，否则下面的断言没有意义
    rc0 = check()
    assert rc0 == 0, "【selftest 无效】注入前基线就不干净（rc=%d），后续断言全是空断言" % rc0
    print("  前提通过：注入前基线干净\n")
    try:
        for name, mutate, expect_field in cases:
            data = json.loads(orig)
            mutate(data["queries"][0])
            with open(GOLDEN, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = check()
            out = buf.getvalue()
            assert rc == 1, "【门失效】%s：注入违规后返回 %d，应为 1" % (name, rc)
            # 前提 2：失败条数恰为 1，且指名被注入的字段
            n_fail = out.count("  FAIL ")
            assert n_fail == 1, "【门不精确】%s：应恰有 1 条失败，实际 %d 条" % (name, n_fail)
            assert expect_field in out, "【门不精确】%s：报错未指名被注入的字段 %s" % (name, expect_field)
            print("  通过：%s 被拦（1 条失败，指名 %s）" % (name, expect_field))
    finally:
        with open(GOLDEN, "w", encoding="utf-8") as f:
            f.write(orig)
    print("SELFTEST OK：4/4 注入违规均被精确拦下，且基线干净")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        rc = check()
        print("\n接地门 %s" % ("OK" if rc == 0 else "FAIL"))
        sys.exit(rc)
