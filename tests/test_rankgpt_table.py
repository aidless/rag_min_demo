#!/usr/bin/env python3
"""B 线 P3 补充项回归测试：RankGPT 列式 rerank + StructGPT 表问答。

覆盖此前「未迁移」的两项，并把可验证的部分冻结为断言：
  - RankGPT：提示完整性 / 排列解析（畸形零崩溃）/ 无 LLM 显式报错 /
    桩函数重排 / 滑窗策略 / 非空性（不同 LLM 输出 → 不同排序）
  - 表问答：EM 归一化 / lookup / count / sum / avg 的确定性答案 / 不可解析 → None

诚实边界（同时被断言）：未注入 llm_fn 时必须抛错，不得静默退化为原序。

运行：python3 -m pytest -q tests/test_rankgpt_table.py
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RR = _load("rankgpt_rerank", "tools/rankgpt_rerank.py")
TQ = _load("table_qa", "tools/table_qa.py")

# ---------------- RankGPT ----------------

def test_prompt_contains_all_passages():
    p = RR.build_listwise_prompt("什么是最好的 RAG", ["甲", "乙", "丙", "丁"])
    for i in (1, 2, 3, 4):
        assert f"[{i}]" in p
    assert "什么是最好的 RAG" in p
    assert "[] > [] > []" in p          # 输出格式契约必须在提示里


def test_parse_permutation_valid():
    assert RR.parse_permutation("[3] > [1] > [2]", 3) == [2, 0, 1]
    assert RR.parse_permutation("2 1 3", 3) == [1, 0, 2]
    assert RR.parse_permutation("1,2,3", 3) == [0, 1, 2]


def test_parse_permutation_malformed_zero_crash():
    bad = [None, "", "   ", "[]", "abc", "[1] > [2]", "[1] > [1] > [2]",
           "[1] > [2] > [4]", "[1] > [2] > [3] > [4]", 12345, ["x"]]
    for b in bad:
        assert RR.parse_permutation(b, 3) is None, f"未拒绝畸形输入: {b!r}"
    # n 非法
    assert RR.parse_permutation("[1]", 0) is None
    assert RR.parse_permutation("[1]", -1) is None


def test_no_llm_raises_not_silent():
    """诚实边界：无 llm_fn 必须显式报错，绝不静默返回原序。"""
    try:
        RR.listwise_rerank("q", ["a", "b"])
        raise AssertionError("未注入 llm_fn 竟然没报错（静默退化）")
    except RR.NoLLMError:
        pass
    try:
        RR.sliding_window_rerank("q", ["a", "b", "c"])
        raise AssertionError("滑窗未注入 llm_fn 竟然没报错")
    except RR.NoLLMError:
        pass


def test_listwise_rerank_with_stub():
    out = RR.listwise_rerank("q", ["a", "b", "c"],
                             llm_fn=lambda q, pr: "[3] > [1] > [2]")
    assert out == ["c", "a", "b"]


def test_nonvacuity_different_llm_output_changes_order():
    """非空性：不同 LLM 输出必须给出不同排序（证明重排真的用了 LLM 输出）。"""
    o1 = RR.listwise_rerank("q", ["a", "b", "c"],
                            llm_fn=lambda q, pr: "[1] > [2] > [3]")
    o2 = RR.listwise_rerank("q", ["a", "b", "c"],
                            llm_fn=lambda q, pr: "[3] > [2] > [1]")
    assert o1 == ["a", "b", "c"]
    assert o2 == ["c", "b", "a"]
    assert o1 != o2


def test_bad_llm_output_not_silently_degraded():
    try:
        RR.listwise_rerank("q", ["a", "b"], llm_fn=lambda q, pr: "I cannot help")
        raise AssertionError("不可解析输出竟然没报错")
    except ValueError:
        pass


def test_sliding_window_runs():
    """滑窗：不丢候选，且至少调用一次 LLM。

    桩按**行首** `[i] ` 识别候选项（提示里内联的格式示例 `[] > [] > []`
    不会被误计），这是正确的解析口径。
    """
    import re
    calls = {"n": 0}

    def stub(q, pr):
        calls["n"] += 1
        ids = re.findall(r"^\[(\d+)\] ", pr, flags=re.M)
        return " > ".join(f"[{i}]" for i in ids)

    out = RR.sliding_window_rerank("q", ["a", "b", "c", "d"], llm_fn=stub,
                                   window=3, step=1)
    assert sorted(out) == ["a", "b", "c", "d"]      # 不丢候选
    assert calls["n"] >= 1


def test_echoed_prompt_is_rejected():
    """守卫：若 LLM 直接把提示回显，数字会重复出现 → 必须被拒（fail-closed），
    而不是解析出一个看似合理的排列。（提示内联的 `[] > [] > []` 是诱因。）"""
    pr = RR.build_listwise_prompt("q", ["a", "b", "c"])
    assert RR.parse_permutation(pr, 3) is None
    try:
        RR.listwise_rerank("q", ["a", "b", "c"], llm_fn=lambda q, p: p)
        raise AssertionError("回显提示竟被接受为排列")
    except ValueError:
        pass


# ---------------- 表问答 ----------------

def test_em_normalization():
    assert TQ.em("2", "2.0")
    assert TQ.em("Beijing", " beijing ")
    assert TQ.em("1,234", "1234")
    assert not TQ.em("x", "y")
    assert not TQ.em(None, "y")


def test_table_answers_frozen():
    t = TQ.load_table()
    cases = {
        "alice 的城市": "beijing",
        "carol 的城市": "beijing",
        "carol 的年限": "5",
        "bob 的城市": "shanghai",
        "RD 有多少人": "3",          # fixture 中 RD 共 3 行
        "Sales 有多少人": "2",
        "salary 总和": "132",        # 30+25+35+20+22
        "平均薪资": "26.4",          # 132/5
        "Sales 的平均薪资": "21",    # (20+22)/2
    }
    for q, gold in cases.items():
        got, plan = TQ.answer(t, q)
        assert TQ.em(got, gold), f"{q}: got={got!r} gold={gold!r} plan={plan}"


def test_table_unparseable_returns_none():
    t = TQ.load_table()
    for q in ("今天天气怎么样", "帮我写首诗", ""):
        got, _ = TQ.answer(t, q)
        assert got is None, f"不可解析查询竟给出答案: {q!r} -> {got!r}"


def test_table_answer_deterministic():
    t = TQ.load_table()
    a = [TQ.answer(t, "salary 总和") for _ in range(3)]
    assert a[0] == a[1] == a[2]
