#!/usr/bin/env python3
"""RankGPT 式列式 rerank（B 线）——补上 P3 未迁移项。

原设计稿的 B 线「RankGPT 列式提示模板」当时未迁（B 线无 LLM/权重）。本模块补上
**可验证的全部非 LLM 部分**，并把 LLM 调用做成**显式注入**，避免假结果：

  1. `build_listwise_prompt`  列式排列生成提示（RankGPT 的核心形式）
  2. `parse_permutation`      排列输出解析（**畸形输入零崩溃**，越界/重复/缺失一律拒绝）
  3. `sliding_window_rerank`  RankGPT 的滑窗策略（窗口内列式重排、逐窗前移）
  4. `listwise_rerank`        可插拔执行器：必须传入 llm_fn

诚实边界：本仓**无 LLM**。`listwise_rerank` 在未注入 `llm_fn` 时**显式报错**，
而不是回退成某个"看起来合理"的排序（那就成了假结果）。
零依赖：标准库 only。
"""

import re

__all__ = [
    "build_listwise_prompt", "parse_permutation",
    "listwise_rerank", "sliding_window_rerank",
]

PROMPT_TEMPLATE = (
    "I will provide you with {n} passages, each indicated by a numerical "
    "identifier []. Rank the passages based on their relevance to the search "
    "query: {query}.\n"
    "{passages}\n"
    "Search Query: {query}.\n"
    "Rank the {n} passages above based on their relevance to the search "
    "query. All passages should be included and listed using identifiers, "
    "in descending order of relevance, in the format of [1] > [2] > [3]. "
    "The output format must be '[] > [] > []', where the number in [] is the "
    "identifier of the passage. Do not explain."
)


class NoLLMError(RuntimeError):
    """未注入 llm_fn 时的显式错误（拒绝伪造排序）。"""


def build_listwise_prompt(query, passages):
    """构造列式排列生成提示。passages 为字符串序列（通常 0-based 序号 1..n）。"""
    lines = [f"[{i + 1}] {p}" for i, p in enumerate(passages)]
    return PROMPT_TEMPLATE.format(n=len(passages), query=query,
                                  passages="\n".join(lines))


def parse_permutation(text, n):
    """解析模型输出的排列；返回 0-based 索引列表，失败返回 None。

    接受 `[1] > [2] > [3]`、`1 2 3`、`1,2,3` 等；要求是 1..n 的一个**完整排列**。
    任何畸形输入（None/非串/空/越界/重复/缺失）一律返回 None —— 零崩溃。
    """
    try:
        if text is None or not isinstance(text, str) or not text.strip():
            return None
        if not isinstance(n, int) or n <= 0:
            return None
        nums = [int(x) for x in re.findall(r"\d+", text)]
        if len(nums) != n:
            return None
        if sorted(nums) != list(range(1, n + 1)):
            return None
        return [x - 1 for x in nums]
    except Exception:
        return None


def listwise_rerank(query, passages, llm_fn=None):
    """列式重排。必须显式注入 llm_fn(query, prompt) -> str。

    返回重排后的 passages。llm_fn 缺失或输出不可解析时**显式报错**，
    绝不静默退化为原序（避免把失败伪装成结果）。
    """
    if llm_fn is None:
        raise NoLLMError(
            "未注入 llm_fn：本仓无 LLM，拒绝伪造排序结果"
            "（如需测试请注入确定性的桩函数）"
        )
    if not passages:
        return []
    prompt = build_listwise_prompt(query, passages)
    raw = llm_fn(query, prompt)
    perm = parse_permutation(raw, len(passages))
    if perm is None:
        raise ValueError("llm_fn 输出无法解析为完整排列（拒绝静默退化）")
    return [passages[i] for i in perm]


def sliding_window_rerank(query, passages, llm_fn=None, window=3, step=1):
    """RankGPT 滑窗策略：自后向前，每窗内做列式重排，再前移 step。

    返回重排后的 passages。窗口过小或候选过少时退化为单窗（但仍需 llm_fn）。
    """
    if llm_fn is None:
        raise NoLLMError("未注入 llm_fn：拒绝伪造排序结果")
    items = list(passages)
    n = len(items)
    if n <= 1:
        return items
    w = max(2, min(int(window), n))
    s = max(1, int(step))
    start = n - w
    while start > 0:
        start = max(0, start)
        seg = items[start:start + w]
        prompt = build_listwise_prompt(query, seg)
        perm = parse_permutation(llm_fn(query, prompt), len(seg))
        if perm is None:
            raise ValueError("滑窗内输出无法解析为完整排列（拒绝静默退化）")
        items[start:start + w] = [seg[i] for i in perm]
        if start == 0:
            break
        start -= s
    if start == 0:
        seg = items[0:w]
        perm = parse_permutation(llm_fn(query, build_listwise_prompt(query, seg)),
                                 len(seg))
        if perm is None:
            raise ValueError("首窗输出无法解析为完整排列（拒绝静默退化）")
        items[0:w] = [seg[i] for i in perm]
    return items
